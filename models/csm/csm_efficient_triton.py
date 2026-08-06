import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from functools import partial
from copy import deepcopy
import time
import math

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

from einops import rearrange, repeat

# triton implements ========================================
@triton.jit
def triton_efficient_cross_3d_scan_flex_o(
        x: tl.tensor,  # (B, C, X, Y, Z) | (B, T, X, Y, Z)
        y: tl.tensor,  # (B, K, C, X, Y, Z) | (B, X, Y, Z, K, C)
        # x_layout: tl.constexpr,
        # y_layout: tl.constexpr,

        operation: tl.constexpr,
        onebyone: tl.constexpr,
        scans: tl.constexpr,
        bidirectional: tl.constexpr,
        K: tl.constexpr,
        BC: tl.constexpr,  # BC
        BX: tl.constexpr,  # BX
        BY: tl.constexpr,  # BY
        BZ: tl.constexpr,  # BZ

        DC: tl.constexpr,  # C
        DX: tl.constexpr,  # T
        DY: tl.constexpr,  # H
        DZ: tl.constexpr,  # W
        NX: tl.constexpr,  # NX
        NY: tl.constexpr,  # NT
        NZ: tl.constexpr,  # NZ
):
    # x_layout = 0
    # y_layout = 1 # 0 BCTHW, 1 BTHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    # 2.调整线程块索引计算
    #  原代码通过 i_yz = program_id(0) 处理空间位置。扩展后需拆分时间索引 i_x：
    i_xyz, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_x = i_xyz // (NY * NZ)  # 时间分块索引
    i_yz = i_xyz % (NY * NZ)  # 空间分块索引（与原逻辑一致）
    i_y, i_z = (i_yz // NZ), (i_yz % NZ)

    # 同时增加时间维度的掩码和位置计算：
    _mask_x = (i_x * BX + tl.arange(0, BX)) < DX  # DT为总时间长度
    _mask_y = (i_y * BY + tl.arange(0, BY)) < DY
    _mask_z = (i_z * BZ + tl.arange(0, BZ)) < DZ

    # 广播掩码维度：
    # _mask_x: (BX, 1, 1)
    # _mask_y: (1, BY, 1)
    # _mask_z: (1, 1, BZ)
    _mask_xyz = _mask_x[:, None, None] & _mask_y[None, :, None] & _mask_z[None, None, :]

    # _mask_hw = _mask_y[:, None] & _mask_z[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_x = (i_x * BX + tl.arange(0, BX)[:, None, None])  # 正向时间位置
    pos_y = (i_y * BY + tl.arange(0, BY)[None, :, None])
    pos_z = (i_z * BZ + tl.arange(0, BZ)[None, None, :])

    neg_x = (DX - i_x * BX - 1 - tl.arange(0, BX)[:, None, None])  # 反向时间位置
    neg_y = (DY - i_y * BY - 1 - tl.arange(0, BY)[None, :, None])
    neg_z = (DZ - i_z * BZ - 1 - tl.arange(0, BZ)[None, None, :])

    if scans == 0:
        '''
        Route 0 : Y*Z面，先扫描一行中的每列（Z轴）,再逐行扫描图片（Y轴），再逐帧堆叠图片（X轴）
        Route 1 : Y*Z面，先扫描一列中的每行（Y轴）,再逐列扫描图片（Z轴），再逐帧堆叠图片（X轴）
        [B, C, X, T, Z] -> [B, K, C, X, Y, Z] -> [B, K, C, X * Y * Z]
        '''
        HWRoute0 = pos_x * DY * DZ + pos_y * DZ + pos_z
        HWRoute1 = pos_x * DZ * DY + pos_z * DY + pos_y
        if bidirectional == 1:
            HWRoute2 = neg_x * DY * DZ + neg_y * DZ + neg_z
            HWRoute3 = neg_x * DZ * DY + neg_z * DY + neg_y
    else:
        raise NotImplementedError

    _tmp1 = DC * DX * DY * DZ
    y_ptr_base = y + i_b * K * _tmp1 + (i_c * BC * DX * DY * DZ)
    p_y1 = y_ptr_base + HWRoute0
    p_y2 = y_ptr_base + _tmp1 + HWRoute1
    p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
    p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3

    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DX * DY * DZ)
        p_x = x_ptr_base + HWRoute0
        if operation == 0:
            for idxc in range(_for_C):
                # _idx_x = idxc * DX * DY * DZ if x_layout == 0 else idxc
                # _idx_y = idxc * DX * DY * DZ if y_layout == 0 else idxc
                _idx_x = idxc * DX * DY * DZ
                _idx_y = idxc * DX * DY * DZ
                _x = tl.load(p_x + _idx_x, mask=_mask_xyz)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_xyz)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_xyz)
                if bidirectional == 1:
                    tl.store(p_y3 + _idx_y, _x, mask=_mask_xyz)
                    tl.store(p_y4 + _idx_y, _x, mask=_mask_xyz)
        elif operation == 1:
            for idxc in range(_for_C):
                # _idx_x = idxc * DX * DY * DZ if x_layout == 0 else idxc
                # _idx_y = idxc * DX * DY * DZ if y_layout == 0 else idxc
                _idx_x = idxc * DX * DY * DZ
                _idx_y = idxc * DX * DY * DZ
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_xyz)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_xyz)
                if bidirectional == 1:
                    _y3 = tl.load(p_y3 + _idx_y, mask=_mask_xyz)
                    _y4 = tl.load(p_y4 + _idx_y, mask=_mask_xyz)
                    tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_xyz)
                else:
                    tl.store(p_x + _idx_x, _y1 + _y2, mask=_mask_xyz)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

@triton.jit
def triton_efficient_cross_3d_scan_flex(
        x: tl.tensor,  # (B, C, X, Y, Z) | (B, T, X, Y, Z)
        y: tl.tensor,  # (B, K, C, X, Y, Z) | (B, X, Y, Z, K, C)
        # x_layout: tl.constexpr,
        # y_layout: tl.constexpr,

        operation: tl.constexpr,
        onebyone: tl.constexpr,
        scans: tl.constexpr,
        bidirectional: tl.constexpr,
        K: tl.constexpr,
        BC: tl.constexpr,  # BC
        BX: tl.constexpr,  # BX
        BY: tl.constexpr,  # BY
        BZ: tl.constexpr,  # BZ

        DC: tl.constexpr,  # C
        DX: tl.constexpr,  # X
        DY: tl.constexpr,  # Y
        DZ: tl.constexpr,  # Z
        NX: tl.constexpr,  # NX
        NY: tl.constexpr,  # NY
        NZ: tl.constexpr,  # NZ
):
    # x_layout = 0
    # y_layout = 1 # 0 BCTHW, 1 BTHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    # 2.调整线程块索引计算
    #  原代码通过 i_yz = program_id(0) 处理空间位置。扩展后需拆分时间索引 i_x：
    i_xyz, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_x = i_xyz // (NY * NZ)  # 时间分块索引
    i_yz = i_xyz % (NY * NZ)  # 空间分块索引（与原逻辑一致）
    i_y, i_z = (i_yz // NZ), (i_yz % NZ)

    # 同时增加时间维度的掩码和位置计算：
    _mask_x = (i_x * BX + tl.arange(0, BX)) < DX  # DT为总时间长度
    _mask_y = (i_y * BY + tl.arange(0, BY)) < DY
    _mask_z = (i_z * BZ + tl.arange(0, BZ)) < DZ

    # 广播掩码维度：
    # _mask_x: (BX, 1, 1)
    # _mask_y: (1, BY, 1)
    # _mask_z: (1, 1, BZ)
    _mask_xyz = _mask_x[:, None, None] & _mask_y[None, :, None] & _mask_z[None, None, :]

    # _mask_hw = _mask_y[:, None] & _mask_z[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_x = (i_x * BX + tl.arange(0, BX)[:, None, None])  # 正向时间位置
    pos_y = (i_y * BY + tl.arange(0, BY)[None, :, None])
    pos_z = (i_z * BZ + tl.arange(0, BZ)[None, None, :])

    neg_x = (DX - i_x * BX - 1 - tl.arange(0, BX)[:, None, None])  # 反向时间位置
    neg_y = (DY - i_y * BY - 1 - tl.arange(0, BY)[None, :, None])
    neg_z = (DZ - i_z * BZ - 1 - tl.arange(0, BZ)[None, None, :])

    # pos_x = (i_x * BX + tl.arange(0, BX)[:, None, None])  # 正向时间位置
    # pos_y = (i_y * BY + tl.arange(0, BY)[None, :, None])
    # pos_z = (i_z * BZ + tl.arange(0, BZ)[None, None, :])
    #
    # neg_x = (DX - i_x * BX - 1 - tl.arange(0, BX)[:, None, None])  # 反向时间位置
    # neg_y = (DY - i_y * BY - 1 - tl.arange(0, BY)[None, :, None])
    # neg_z = (DZ - i_z * BZ - 1 - tl.arange(0, BZ)[None, None, :])

    if scans == 0:
        '''
        Route 0 : Y*Z面，先扫描一行中的每列（Z轴）,再逐行扫描图片（Y轴），再逐帧堆叠图片（X轴）
        Route 1 : Y*Z面，先扫描一列中的每行（Y轴）,再逐列扫描图片（Z轴），再逐帧堆叠图片（X轴）
        [B, C, X, Y, Z] -> [B, K, C, X, Y, Z] -> [B, K, C, X * Y * Z]
        '''
        HWRoute0 = pos_x * DY * DZ + pos_y * DZ + pos_z
        HWRoute1 = pos_x * DZ * DY + pos_z * DY + pos_y
        if bidirectional == 1:
            HWRoute2 = neg_x * DY * DZ + neg_y * DZ + neg_z
            HWRoute3 = neg_x * DZ * DY + neg_z * DY + neg_y
    else:
        raise NotImplementedError

    _tmp1 = (DC * DX * DY * DZ) // 4
    y_ptr_base = y + i_b * K * _tmp1 + (i_c * BC * DX * DY * DZ // 4)
    p_y1 = y_ptr_base + HWRoute0
    p_y2 = y_ptr_base + _tmp1 + HWRoute1
    p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
    p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3

    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DX * DY * DZ // 4)
        p_x = x_ptr_base + HWRoute0
        if operation == 0:
            for idxc in range(_for_C):
                # _idx_x = idxc * DX * DY * DZ if x_layout == 0 else idxc
                # _idx_y = idxc * DX * DY * DZ if y_layout == 0 else idxc
                _idx_x = idxc * DX * DY * DZ // 4
                _idx_y = idxc * DX * DY * DZ // 4
                _x = tl.load(p_x + _idx_x, mask=_mask_xyz)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_xyz)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_xyz)
                if bidirectional == 1:
                    tl.store(p_y3 + _idx_y, _x, mask=_mask_xyz)
                    tl.store(p_y4 + _idx_y, _x, mask=_mask_xyz)
        elif operation == 1:
            for idxc in range(_for_C):
                # _idx_x = idxc * DX * DY * DZ if x_layout == 0 else idxc
                # _idx_y = idxc * DX * DY * DZ if y_layout == 0 else idxc
                _idx_x = idxc * DX * DY * DZ
                _idx_y = idxc * DX * DY * DZ
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_xyz)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_xyz)
                if bidirectional == 1:
                    _y3 = tl.load(p_y3 + _idx_y, mask=_mask_xyz)
                    _y4 = tl.load(p_y4 + _idx_y, mask=_mask_xyz)
                    tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_xyz)
                else:
                    tl.store(p_x + _idx_x, _y1 + _y2, mask=_mask_xyz)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


class EfficientCross3DScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, one_by_one=False, scans=0, bidirectional=True, step_size=2, padded=False):
        if one_by_one:
            B, _, C, X, ori_Y, ori_Z = x.shape
        else:
            B, C, X, ori_Y, ori_Z = x.shape
        B, C, X, ori_Y, ori_Z = int(B), int(C), int(X), int(ori_Y), int(ori_Z)

        if not padded:
            if ori_Y % step_size != 0:
                pad_Y = step_size - ori_Y % step_size
                x = F.pad(x, (0, pad_Y, 0, 0))
            Y = x.shape[-1]

            if ori_Z % step_size != 0:
                pad_Z = step_size - ori_Z % step_size
                x = F.pad(x, (0, 0, 0, pad_Z))
            Z = x.shape[-2]

        K = 4 if bidirectional else 2
        # 1. 修改网格配置
        # 原网格配置为(NY * NZ, NC, B)，表示对空间维度（Y, Z）分块。扩展时间维度后，需将时间分块 NX 加入：
        BX, BY, BZ, BC = 4, 8, 8, 1
        NX, NY, NZ, NC = triton.cdiv(X, BX), triton.cdiv(Y, BY), triton.cdiv(Z, BZ), triton.cdiv(C, BC)

        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.bidirectional = bidirectional
        ctx.ori_shape = (B, C, X, ori_Y, ori_Z)
        ctx.K = K
        ctx.expand_routes = K
        ctx.triton_shape = (BC, BX, BY, BZ, NC, NX, NY, NZ)

        y = x.new_empty((B, K, C, X, Y // step_size, Z // step_size))
        triton_efficient_cross_3d_scan_flex[(NX * NY * NZ, NC, B)](
            x.contiguous(), y, 0, (0 if not one_by_one else 1), scans, (0 if not bidirectional else 1),
            K, BC, BX, BY, BZ, C, X, Y, Z, NX, NY, NZ
        )
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        bidirectional = ctx.bidirectional
        step_size = ctx.step_size
        B, C, X, ori_Y, ori_Z = ctx.ori_shape
        K = ctx.K
        BC, BX, BY, BZ, NC, NX, NY, NZ = ctx.triton_shape

        Y, Z = math.ceil(ori_Y / step_size), math.ceil(ori_Z / step_size)
        new_Y, new_Z = Y * step_size, Z * step_size

        if one_by_one:
            x = y.new_empty((B, K, C, X, new_Y, Z))
        else:
            x = y.new_empty((B, C, X, new_Y, Z))

        triton_efficient_cross_3d_scan_flex[(NX * NY * NZ, NC, B)](
            x, y.contiguous(), 1, (0 if not one_by_one else 1), scans, (0 if not bidirectional else 1),
            K, BC, BX, BY, BZ, C, X, new_Y, Z, NX, NY, NZ
        )
        return x, None, None, None, None, None, None


class EfficientCross3DMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, one_by_one=False, scans=0, bidirectional=True):
        B, K, C, X, Y, Z = y.shape
        B, K, C, X, Y, Z = int(B), int(K), int(C), int(X), int(Y), int(Z)
        # K = 4 if bidirectional else 2
        # 1. 修改网格配置
        # 原网格配置为(NY * NZ, NC, B)，表示对空间维度（Y, Z）分块。扩展时间维度后，需将时间分块 NX 加入：
        BX, BY, BZ, BC = 1, 16, 16, 1
        NX, NY, NZ, NC = triton.cdiv(X, BX), triton.cdiv(Y, BY), triton.cdiv(Z, BZ), triton.cdiv(C, BC)

        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.bidirectional = bidirectional
        # ctx.scan_fn = scan_fn
        ctx.shape = (B, C, X, Y, Z)
        ctx.K = K
        ctx.triton_shape = (BC, BX, BY, BZ, NC, NX, NY, NZ)
        ctx.expand_routes = K

        # if one_by_one:
        #     x = y.new_empty((B, K, C, X * Y * Z))
        # else:
        #     x = y.new_empty((B, C, X * Y * Z))
        if one_by_one:
            x = y.new_empty((B, K, C, X, Y, Z))
        else:
            x = y.new_empty((B, C, X, Y, Z))

        triton_efficient_cross_3d_scan_flex[(NX * NY * NZ, NC, B)](
            x, y.contiguous(), 1, (0 if not one_by_one else 1), scans, (0 if not bidirectional else 1),
            K, BC, BX, BY, BZ, C, X, Y, Z, NX, NY, NZ
        )
        return x

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        bidirectional = ctx.bidirectional
        B, C, X, Y, Z = ctx.shape
        BC, BX, BY, BZ, NC, NX, NY, NZ = ctx.triton_shape
        K = ctx.K
        y = x.new_empty((B, K, C, X, Y, Z))

        triton_efficient_cross_3d_scan_flex[(NX * NY * NZ, NC, B)](
            x.contiguous(), y, 0, (0 if not one_by_one else 1), scans, (0 if not bidirectional else 1),
            K, BC, BX, BY, BZ, C, X, Y, Z, NX, NY, NZ
        )
        return y, None, None, None, None, None


# Selector ========================================
class EfficientRouteSelector(nn.Module):
    def __init__(self, route_type: str='spatial', bidirectional: bool=True, merge_mean: bool=True):
        '''
        route_type: 选择扫描路径  's' 'spatial';      't' 'temporal';   'st' 'spatiotemporal';   'sy' 'synthetic'
        step_size: 间隔长度，当 type=='efficient'时生效
        bidirectional: 是否反向
        '''
        super(EfficientRouteSelector, self).__init__()

        self.bidirectional = bidirectional
        self.merge_mean = merge_mean

        if route_type in ['sy', 'synthetic']:
            self.scan = self._scan_synthetic
            self.merge = self._merge_synthetic
            self.K = 12 if bidirectional else 8

        elif route_type in ['s', 'spatial']:
            self.scan = self._scan_spatial
            self.merge = self._merge_spatial
            self.K = 4 if bidirectional else 2

        elif route_type in ['t', 'temporal']:
            self.scan = self._scan_temporal
            self.merge = self._merge_temporal
            self.K = 4 if bidirectional else 2

        elif route_type in ['st', 'spatiotemporal']:
            self.scan = self._scan_spatiotemporal
            self.merge = self._merge_spatiotemporal
            self.K = 4 if bidirectional else 2
        else:
            raise NotImplementedError(f'route_type {route_type} not implemented')

        if WITH_TRITON:
            self.CSF = EfficientCross3DScanTritonF
            self.CMF = EfficientCross3DMergeTritonF
        else:
            raise NotImplementedError

    ### ========== Spatial =======================================================
    def _scan_spatial(self, x: torch.Tensor, step_size=2, padded=False):
        '''
        Route 1 (Spatial): H*W面，先扫描一行中的每列（W轴）,再逐行扫描图片（H轴），再逐帧堆叠图片（T轴）
        Route 2 (Spatial): H*W面，先扫描一列中的每行（H轴）,再逐列扫描图片（W轴），再逐帧堆叠图片（T轴）
        [B, C, T, H, W] -> [B, K, C, T, H, W] -> [B, K, C, T * H * W]
        '''
        # K = 4 if self.bidirectional else 2
        self.ori_shape = [B, C, ori_T, ori_H, ori_W] = x.shape    # [B, C, T, H, W]

        # if not padded:
        #     if ori_W % step_size != 0:
        #         pad_W = step_size - ori_W % step_size
        #         x = F.pad(x, (0, pad_W, 0, 0))
        #     W = x.shape[-1]
        #
        #     if ori_H % step_size != 0:
        #         pad_H = step_size - ori_H % step_size
        #         x = F.pad(x, (0, 0, 0, pad_H))
        #     H = x.shape[-2]
        print(x)
        with torch.cuda.device(x.device):
            y = self.CSF.apply(x, False, 0, self.bidirectional, step_size, padded)
        self.tar_shape = [B, K, C, T, H, W] = y.shape  # [B, K, C, T, H, W]
        y = y.view(B, K, C, T * H * W)   # [B, K, C, T * H * W]
        return y


    def _merge_spatial(self, y: torch.Tensor):
        '''
        [B, K, C, T * H * W] -> [B, K, C, T, H, W] ->  [B, C, T, H, W]
        '''
        y = y.view(self.tar_shape)   # [B, K, C, T, H, W]
        with torch.cuda.device(y.device):
            x = self.CMF.apply(y, False, 0, self.bidirectional)
        if self.merge_mean:
            K = y.shape[1]
            x = x / float(K)
        return x  #  [B, C, T, H, W]

    ### ========== Temporal =======================================================
    def _scan_temporal(self, x: torch.Tensor):
        '''
        Route 3 (Temporal) :       T*W面，先逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴） 再逐行扫描图片（H轴）
        Route 5 (Spatiotemporal) : T*W面，先再扫描一行中的每列（W轴），逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）
        [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T, W] -> [B, K, C, H * T * W]
        '''
        # K = 4 if self.bidirectional else 2
        x = x.transpose(dim0=-2, dim1=-3)  # [B, C, T, H, W] ->  [B, C, H, T, W]
        self.ori_shape = x.shape  # [B, C, H, T, W]
        with torch.cuda.device(x.device):
            y = self.CSF.apply(x, False, 0, self.bidirectional)
        self.tar_shape = [B, K, C, H, T, W] = y.shape  # [B, K, C, H, T, W]
        y = y.view(B, K, C, H * T * W)  # [B, K, C, H * T * W]
        return y

    def _merge_temporal(self, y: torch.Tensor):
        '''
        [B, K, C, H * T * W] -> [B, K, C, H, T, W] ->  [B, C, H, T, W] -> [B, C, T, H, W]
        '''
        y = y.view(self.tar_shape)  # [B, K, C, T, H, W]
        with torch.cuda.device(y.device):
            x = self.CMF.apply(y, False, 0, self.bidirectional)  # [B, C, H, T, W]
        x = x.transpose(dim0=-2, dim1=-3)  # [B, C, H, T, W] ->  [B, C, T, H, W]
        if self.merge_mean:
            K = y.shape[1]
            x = x / float(K)
        return x

    ### ========== Spatiotemporal =======================================================
    def _scan_spatiotemporal(self, x: torch.Tensor):
        '''
        Route 4 (Temporal) : T*H面，先逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）,再扫描一行中的每列（W轴）
        Route 6 (Spatiotemporal) : T*H面，先逐行扫描图片（H轴）,再逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴）
        [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T, H] -> [B, K, C, W * T * H]
        '''
        # K = 4 if self.bidirectional else 2
        x = x.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W] ->  [B, C, W, T, H]
        self.ori_shape = x.shape  # [B, C, W, T, H]

        with torch.cuda.device(x.device):
            y = self.CSF.apply(x, False, 0, self.bidirectional)
        self.tar_shape = [B, K, C, W, T, H] = y.shape  # [B, K, C, W, T, H]
        y = y.view(B, K, C, W * T * H) # [B, K, C, W * T * H]
        return y

    def _merge_spatiotemporal(self, y: torch.Tensor):
        '''
        [B, K, C, W * T * H] -> [B, K, C, W, T, H] ->  [B, C, W, T, H] ->  [B, C, T, H, W]
        '''
        y = y.view(self.tar_shape)   # [B, K, C, W, T/s, H/s]
        with torch.cuda.device(y.device):
            x = self.CMF.apply(y, False, 0, self.bidirectional) #   [B, C, W, T, H]
        x = x.permute(0, 1, 3, 4, 2)  # [B, C, T, H, W]
        if self.merge_mean:
            K = y.shape[1]
            x = x / float(K)
        return x

    ### ========== Spatiotemporal =======================================================
    def _scan_synthetic(self, x: torch.Tensor):
        '''
        Route 1 (Spatial)  Route 2 (Spatial):
        [B, C, T, H, W] -> [B, K, C, T, H, W] -> [B, K, C, T * H * W]

        Route 3 (Temporal) Route 5 (Spatiotemporal) :
        [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T, W] -> [B, K, C, H * T * W]

        Route 4 (Temporal) Route 6 (Spatiotemporal) :
        [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T, H] -> [B, K, C, W * T * H]

        Return:  [Route 1, Route 2, Route 3, Route 5, Route 4, Route 6]   [B, K', C, H, W, T * H * W]
        '''
        self.tar_shape_list = []
        y_spatial = self._scan_spatial(x)
        self.tar_shape_list.append(self.tar_shape)

        y_temporal = self._scan_temporal(x)
        self.tar_shape_list.append(self.tar_shape)

        y_spatiotemporal = self._scan_spatiotemporal(x)
        self.tar_shape_list.append(self.tar_shape)
        y = torch.cat([y_spatial, y_temporal, y_spatiotemporal], dim=1)
        return y

    def _merge_synthetic(self, ys: torch.Tensor):
        ys_spatial, ys_temporal, ys_spatiotemporal = torch.chunk(ys, 3, dim=1)
        self.tar_shape = self.tar_shape_list[0]
        x_spatial = self._merge_spatial(ys_spatial)

        self.tar_shape = self.tar_shape_list[1]
        x_temporal = self._merge_temporal(ys_temporal)

        self.tar_shape = self.tar_shape_list[2]
        x_spatiotemporal = self._merge_spatiotemporal(ys_spatiotemporal)


        x = torch.stack([x_spatial, x_temporal, x_spatiotemporal], dim=1)

        x = torch.mean(x, dim=1) if self.merge_mean else torch.sum(x, dim=1)
        return x




# checks =================================================================



def check_efficient_cms_triton_scan():
    def efficient_cross_scan(x: torch.Tensor, route_type, step_size=2):
        B, C, ori_T, ori_H, ori_W = x.shape

        if ori_W % step_size != 0:
            pad_W = step_size - ori_W % step_size
            x = F.pad(x, (0, pad_W, 0, 0))
        W = x.shape[-1]

        if ori_H % step_size != 0:
            pad_H = step_size - ori_H % step_size
            x = F.pad(x, (0, 0, 0, pad_H))
        H = x.shape[-2]

        if ori_T % step_size != 0:
            pad_T = step_size - ori_T % step_size
            x = F.pad(x, (0, 0, 0, 0, 0, pad_T))
        T = x.shape[-3]


        # T, H, W  = T // step_size, H // step_size, W // step_size
        # L = ori_T * ori_H * ori_W // (step_size ** 2)

        # print(x)
        r0 = x[..., ::step_size, ::step_size].contiguous().view(B, C, -1)
        r1 = x.transpose(-1, -2)[..., ::step_size, 1::step_size].contiguous().view(B, C, -1)
        r2 = x[..., ::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])
        r3 = x.transpose(-1, -2)[..., 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])
        # print(x.transpose(-2, -3))
        r4 = x.transpose(-2, -3)[..., ::step_size, ::step_size].contiguous().view(B, C, -1)  # [B, C, T, H, W] ->  [B, C, H, T, W] ->  [B, C, H * T * W]
        r5 = x.transpose(-2, -3).transpose(-1, -2)[..., ::step_size, 1::step_size].contiguous().view(B, C, -1)
        r6 = x.transpose(-2, -3)[..., ::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])
        r7 = x.transpose(-2, -3).transpose(-1, -2)[..., 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])
        # print(x.permute(0, 1, 4, 2, 3))
        r8 = x.permute(0, 1, 4, 2, 3)[..., ::step_size, ::step_size].contiguous().view(B, C, -1)
        r9 = x.permute(0, 1, 4, 2, 3).transpose(-1, -2)[..., ::step_size, 1::step_size].contiguous().view(B, C, -1)
        r10 = x.permute(0, 1, 4, 2, 3)[..., ::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])
        r11 = x.permute(0, 1, 4, 2, 3).transpose(-1, -2)[..., 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip(dims=[-1])

        # r0 = x.view(B, C, L)    # [B, C, T, H, W] ->  [B, C, T * H * W]
        # r1 = x.transpose(-1, -2).contiguous().view(B, C, L)
        # r2 = x.contiguous().view(B, C, L).flip(dims=[-1])
        # r3 = x.transpose(-1, -2).contiguous().view(B, C, L).flip(dims=[-1])
        #
        # r4 = x.transpose(-2, -3).contiguous().view(B, C, L)   # [B, C, T, H, W] ->  [B, C, H, T, W] ->  [B, C, H * T * W]
        # r5 = x.transpose(-2, -3).transpose(-1, -2).contiguous().view(B, C, L)
        # r6 = x.transpose(-2, -3).contiguous().view(B, C, L).flip(dims=[-1])
        # r7 = x.transpose(-2, -3).transpose(-1, -2).contiguous().view(B, C, L).flip(dims=[-1])
        #
        # r8 = x.permute(0, 1, 4, 2, 3).contiguous().view(B, C, L)  # [B, C, T, H, W] ->  [B, C, W, T, H] ->  [B, C, H * T * W]
        # r9 = x.permute(0, 1, 4, 2, 3).transpose(-1, -2).contiguous().view(B, C, L)
        # r10 = x.permute(0, 1, 4, 2, 3).contiguous().view(B, C, L).flip(dims=[-1])
        # r11 = x.permute(0, 1, 4, 2, 3).transpose(-1, -2).contiguous().view(B, C, L).flip(dims=[-1])

        if route_type in ['s', 'spatial']:
            route_list = [r0, r1, r2, r3]
        elif route_type in ['t', 'temporal']:
            route_list = [r4, r5, r6, r7]
        elif route_type in ['st', 'spatiotemporal']:
            route_list = [r8, r9, r10, r11]
        elif route_type in ['sy', 'synthetic']:
            route_list = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]
        else:
            raise NotImplementedError

        xs = torch.stack(route_list, dim = 1)
        # K = len(route_list)
        return xs

    def efficient_cross_merge(y: torch.Tensor, route_type, ori_shape, merge_mean=True, step_size=2):
        B, C, ori_T, ori_H, ori_W = ori_shape
        # L = T * H * W
        T, H, W = math.ceil(ori_T / step_size), math.ceil(ori_H / step_size), math.ceil(ori_W / step_size)
        new_T, new_H, new_W = T * step_size, H * step_size, W * step_size

        # y1, y2, y3 = y.new_empty((B, C, new_T, new_H, new_W)), y.new_empty((B, C, new_T, new_H, new_W)), y.new_empty((B, C, new_T, new_H, new_W))
        y1, y2, y3 = y.new_empty((B, C, new_T, new_H, new_W)), y.new_empty((B, C,  new_H, new_T,new_W)), y.new_empty((B, C, new_W, new_T, new_H))

        y1[..., ::step_size, ::step_size] = y[:, 0].reshape(B, C, new_T, H, W)
        y1[..., 1::step_size, ::step_size] = y[:, 1].reshape(B, C, new_T, W, H).transpose(-1, -2)
        y1[..., ::step_size, 1::step_size] = y[:, 2].reshape(B, C, new_T, H, W).flip(dims=[-3,-2,-1])
        y1[..., 1::step_size, 1::step_size] = y[:, 3].reshape(B, C, new_T, W, H).transpose(-1, -2).flip(dims=[-3,-2,-1])
        print(y1)
        y2[..., ::step_size, ::step_size] = y[:, 4].view(B, C, new_H, T, W).contiguous()
        y2[..., 1::step_size, ::step_size] = y[:, 5].view(B, C, new_H, W, T).transpose(-1, -2).contiguous()
        y2[..., ::step_size, 1::step_size] = y[:, 6].view(B, C, new_H, T, W).contiguous().flip(dims=[-3,-2,-1])
        y2[..., 1::step_size, 1::step_size] = y[:, 7].view(B, C, new_H, W, T).transpose(-1, -2).contiguous().flip(dims=[-3,-2,-1])
        y2 = y2.transpose(-2, -3)
        print(y2)
        y3[..., ::step_size, ::step_size] = y[:, 8].view(B, C, new_W, T, H).contiguous()
        y3[..., 1::step_size, ::step_size] = y[:, 9].view(B, C, new_W, H, T).transpose(-1, -2).contiguous()
        y3[..., ::step_size, 1::step_size] = y[:, 10].view(B, C, new_W, T, H).contiguous().flip(dims=[-3,-2,-1])
        y3[..., 1::step_size, 1::step_size] = y[:, 11].view(B, C, new_W, H, T).transpose(-1, -2).contiguous().flip(dims=[-3,-2,-1])
        y3 = y3.permute(0, 1, 3, 4, 2)
        print(y3)
        # r0_b = y[:, 0].view(B, C, T, H, W)
        # r1_b = y[:, 1].view(B, C, T, H, W).transpose(-1, -2).contiguous()
        # r2_b = y[:, 2].view(B, C, T, H, W).contiguous().flip(dims=[-3,-2,-1])
        # r3_b = y[:, 3].view(B, C, T, H, W).transpose(-1, -2).contiguous().flip(dims=[-3,-2,-1])
        #
        # r4_b = y[:, 4].view(B, C, H, T, W).transpose(-2, -3).contiguous()
        # r5_b = y[:, 5].view(B, C, H, W, T).transpose(-1, -2).transpose(-2, -3).contiguous()
        # r6_b = y[:, 6].view(B, C, H, T, W).transpose(-2, -3).contiguous().flip(dims=[-3,-2,-1])
        # r7_b = y[:, 7].view(B, C, H, W, T).transpose(-1, -2).transpose(-2, -3).contiguous().flip(dims=[-3,-2,-1])
        #
        # r8_b = y[:, 8].view(B, C, W, T, H).permute(0, 1, 3, 4, 2).contiguous()
        # r9_b = y[:, 9].view(B, C, W, H, T).transpose(-1, -2).permute(0, 1, 3, 4, 2).contiguous()
        # r10_b = y[:, 10].view(B, C, W, T, H).permute(0, 1, 3, 4, 2).contiguous().flip(dims=[-3,-2,-1])
        # r11_b = y[:, 11].view(B, C, W, H, T).transpose(-1, -2).permute(0, 1, 3, 4, 2).contiguous().flip(dims=[-3,-2,-1])

        if ori_T != new_T or ori_H != new_H or ori_W != new_W:
            y1 = y1[..., :ori_T, :ori_H, :ori_W].contiguous()
            y2 = y2[..., :ori_T, :ori_H, :ori_W].contiguous()
            y3 = y3[..., :ori_T, :ori_H, :ori_W].contiguous()

        if route_type in ['s', 'spatial']:
            return y1
        elif route_type in ['t', 'temporal']:
            return y2
        elif route_type in ['st', 'spatiotemporal']:
            return y3
        elif route_type in ['sy', 'synthetic']:
            ys = torch.stack([y1, y2, y3], dim=1)
            ys = ys.mean(dim=1) if merge_mean else ys.sum(dim=1)
            return ys
        else:
            raise NotImplementedError


    route_type = 's'
    if route_type in ['sy', 'synthetic']:
        K = 12
    elif route_type in ['s', 'spatial', 't', 'temporal', 'st', 'spatial']:
        K = 4
    else:
        raise NotImplementedError

    B, T, H, W, C = 1, 4, 4, 4, 1
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    x = torch.zeros([B, T, H, W, C], device=device)
    x = x.view(B, -1, C)
    for i in range(T * H * W):
        x[:, i, :] = i + 1
    x = x.view(B, C, T, H, W).requires_grad_(True)  # channel first
    print(x)
    print(x.shape)
    x1 = x.clone().detach().requires_grad_(True)

    ori_shape = (B, C, T, H, W)
    y = efficient_cross_scan(x, 'sy').clone().detach().requires_grad_(True)
    # y = torch.randn((B, 12, C, T, H, W), device=device).requires_grad_(True)
    y1 = y.clone().detach().requires_grad_(True)

    merge_mean = False
    RS = EfficientRouteSelector(route_type, bidirectional=True, merge_mean=merge_mean)
    cross_3d_scan_fn = RS.scan
    cross_3d_merge_fn = RS.merge

    print('Forward test ------')
    s_time = time.time()
    cs_res_s = triton.testing.do_bench(lambda: efficient_cross_scan(x, route_type))
    t_time = time.time()
    cs_res_s1 = triton.testing.do_bench(lambda: cross_3d_scan_fn(x))
    e_time = time.time()
    print(f'cs_res_s:{cs_res_s}, execution time:{t_time - s_time} seconds')
    print(f'cs_res_s1:{cs_res_s1}, execution time:{e_time - t_time} seconds')


    s_time = time.time()
    cm_res_s = triton.testing.do_bench(lambda: efficient_cross_merge(y, route_type, ori_shape, merge_mean))
    # t_time = time.time()
    # cm_res_s1 = triton.testing.do_bench(lambda: cross_3d_merge_fn(y))
    # e_time = time.time()
    # print(f'cm_res_s:{cm_res_s}, execution time:{t_time - s_time:3f} seconds')
    # print(f'cm_res_s1:{cm_res_s1}, execution time:{e_time - t_time:3f} seconds')


    print('Backward test ------')
    s_time = time.time()
    b_cs_res_s = triton.testing.do_bench(lambda: efficient_cross_scan(x, route_type).sum().backward())
    t_time = time.time()
    b_cs_res_s1 = triton.testing.do_bench(lambda: cross_3d_scan_fn(x).sum().backward())
    e_time = time.time()
    print(f'b_cs_res_s:{b_cs_res_s}, execution time:{t_time - s_time:3f} seconds')
    print(f'b_cs_res_s1:{b_cs_res_s1}, execution time:{e_time - t_time:3f} seconds')

    s_time = time.time()
    b_cm_res_s = triton.testing.do_bench(lambda: efficient_cross_merge(y, route_type, ori_shape, merge_mean).sum().backward())
    t_time = time.time()
    b_cm_res_s1 = triton.testing.do_bench(lambda: cross_3d_merge_fn(y).sum().backward())
    e_time = time.time()
    print(f'b_cm_res_s:{b_cm_res_s}, execution time:{t_time - s_time:3f} seconds')
    print(f'b_cm_res_s1:{b_cm_res_s1}, execution time:{e_time - t_time:3f} seconds')


    print('Grad test ------')
    x.grad, x1.grad = None, None
    y.grad, y1.grad = None, None

    o0 = efficient_cross_scan(x, route_type)
    o1 = cross_3d_scan_fn(x1)

    print(f'o0:{o0} \n o0.shape:{o0.shape}')
    print(f'o1:{o1} \n o1.shape:{o1.shape}')

    o0.backward(y.view(B, K, C, T * H * W))
    o1.backward(y.view(B, K, C, T * H * W))

    print((o0 - o1).abs().max())
    print((x.grad - x1.grad).abs().max())

    o0 = efficient_cross_merge(y, route_type, ori_shape, merge_mean)
    o1 = cross_3d_merge_fn(y1)
    o0.backward(x.view(B, C, T, H, W))
    o1.backward(x.view(B, C, T, H, W))
    print((o0 - o1).abs().max())
    print((y.grad - y1.grad).abs().max())
    x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
    print("===============", flush=True)

if __name__ == "__main__":
    check_efficient_cms_triton_scan()




