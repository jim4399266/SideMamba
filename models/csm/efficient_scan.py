from typing import Any

import torch
import warnings
import math
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from timm.models.byobnet import repvgg_a1
from typing import Union, List, Tuple

###  ====================================== BaseFn =================================================
###  ===========================================================================================================
class EfficientScanFn(torch.autograd.Function):
    # 输入维度是： [B, C, X, Y, Z] ，对 Y , Z 维度进行扫描， 在 X 维度进行堆叠，就不说 T, H, W了
    # @staticmethod
    def forward(ctx, x: torch.Tensor, step_size: int=2, bidirectional: bool=True):
        # [B, C, X, Y, Z] -> [B, K, C, X, Y/s, Z/s]
        # x 为待扫描的张量，已完成padding（如需）
        B, C, X, Y, Z = x.shape
        K = 4 if bidirectional else 2

        ctx.ori_shape = [B, C, X, Y, Z]
        ctx.step_size = step_size
        ctx.bidirectional = bidirectional

        # 填充
        if Z % step_size != 0:
            pad_Z = step_size - Z % step_size
            x = F.pad(x, (0, pad_Z, 0, 0))
        pad_Z = x.shape[-1]     # 填充后的Z长度

        if Y % step_size != 0:
            pad_Y = step_size - Y % step_size
            x = F.pad(x, (0, 0, 0, pad_Y))
        pad_Y = x.shape[-2]     # 填充后的Y长度

        spl_Y, spl_Z = pad_Y // step_size, pad_Z // step_size   # spl_ 表示切分后的目标张量维度

        # spl_Y, spl_Z = Y // step_size, Z // step_size   # spl_ 表示切分后的目标张量维度
        # assert spl_Y * step_size == Y   # 确保可以整分
        # assert spl_Z * step_size == Z


        xs = x.new_empty((B, K, C, X, spl_Y, spl_Z))

        xs[:, 0] = x[:, :, :, ::step_size, ::step_size].contiguous().view(B, C, X, spl_Y, spl_Z)
        xs[:, 1] = x.transpose(dim0=-1, dim1=-2)[:, :, :, ::step_size, 1::step_size].contiguous().view(B, C, X, spl_Y, spl_Z)
        if bidirectional:
            xs[:, 2] = x[:, :, :, ::step_size, 1::step_size].contiguous().view(B, C, X, spl_Y, spl_Z).flip([-1, -2, -3])
            xs[:, 3] = x.transpose(dim0=-1, dim1=-2)[:, :, :, 1::step_size, 1::step_size].contiguous().view(B, C, X, spl_Y, spl_Z).flip(
                [-1, -2, -3])

        return xs  #  [B, K, C, X, Y/s, Z/s]

    # @staticmethod
    def backward(ctx, grad_xs: torch.Tensor): #  [B, K, C, X, Y/s, Z/s]  -> [B, C, X, Y, Z]
        B, C, X, Y, Z = ctx.ori_shape
        step_size = ctx.step_size
        bidirectional = ctx.bidirectional

        spl_Y, spl_Z = grad_xs.shape[-2], grad_xs.shape[-1]  # 可能带padding

        # spl_Y, spl_Z = Y // step_size, Z // step_size  # spl_ 表示切分后的目标张量维度

        grad_x = grad_xs.new_empty((B, C, X, spl_Y * step_size, spl_Z * step_size))

        grad_x[:, :, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, X, spl_Y, spl_Z)
        grad_x[:, :, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, X, spl_Z, spl_Y).transpose(dim0=-1,
                                                                                                            dim1=-2)
        if bidirectional:
            grad_x[:, :, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, X, spl_Y, spl_Z).flip([-1, -2, -3])
            grad_x[:, :, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, X, spl_Z, spl_Y).transpose(
                dim0=-1, dim1=-2).flip([-1, -2, -3])

        if Y != grad_x.shape[-2] or Z != grad_x.shape[-1]:   # 去除padding
            grad_x = grad_x[:, :, :, :Y, :Z]

        return grad_x, None, None  # [B, C, X, Y, Z]

class EfficientMergeFn(torch.autograd.Function):
    # [B, K, C, X, Y/s, Z/s]
    # @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_shape, step_size: int=2, bidirectional: bool=True): # [B, K, C, X, Y/s, Z/s]  -> [B, C, X, Y, Z]
        B, K, C, X, spl_Y, spl_Z = ys.shape
        Y, Z = ori_shape[-2:]
        ctx.spl_shape = [B, K, C, X, spl_Y, spl_Z]
        ctx.step_size = step_size
        ctx.bidirectional = bidirectional

        mer_Y, mer_Z = spl_Y * step_size, spl_Z * step_size

        y = ys.new_empty((B, C, X, mer_Y, mer_Z))

        y[:, :, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, X, spl_Y, spl_Z)
        y[:, :, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, X, spl_Z, spl_Y).transpose(dim0=-1, dim1=-2)
        if bidirectional:
            y[:, :, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, X, spl_Y, spl_Z).flip([-1, -2, -3])
            y[:, :, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, X, spl_Z, spl_Y).transpose(dim0=-1, dim1=-2).flip(
                [-1, -2, -3])

        if Y != mer_Y or Z != mer_Z:   # 去除padding
            y = y[:, :, :, :Y, :Z].contiguous()

        return y  # [B, C, X, Y, Z]

    # @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # [B, C, X, Y, Z] -> [B, K, C, X, Y/s, Z/s]
        Y, Z = grad_y.shape[-2:]
        B, K, C, X, spl_Y, spl_Z = ctx.spl_shape
        step_size = ctx.step_size
        bidirectional = ctx.bidirectional

        # grad_y = grad_y.view(B, C, X, spl_Y * step_size, spl_Z * step_size)
        # 填充
        if Z % step_size != 0:
            pad_Z = step_size - Z % step_size
            grad_y = F.pad(grad_y, (0, pad_Z, 0, 0))
        pad_Z = grad_y.shape[-1]  # 填充后的Z长度

        if Y % step_size != 0:
            pad_Y = step_size - Y % step_size
            grad_y = F.pad(grad_y, (0, 0, 0, pad_Y))
        pad_Y = grad_y.shape[-2]  # 填充后的Y长度
        #
        spl_Y, spl_Z = pad_Y // step_size, pad_Z // step_size   # spl_ 表示切分后的目标张量维度

        grad_ys = grad_y.new_empty((B, K, C, X, spl_Y, spl_Z))

        grad_ys[:, 0] = grad_y[:, :, :, ::step_size, ::step_size].contiguous().view(B, C, X, spl_Y, spl_Z)
        grad_ys[:, 1] = grad_y.transpose(dim0=-1, dim1=-2)[:, :, :, ::step_size, 1::step_size].contiguous().view(B, C,
                                                                                                                 X, spl_Y, spl_Z)
        if bidirectional:
            grad_ys[:, 2] = grad_y[:, :, :, ::step_size, 1::step_size].contiguous().view(B, C, X, spl_Y, spl_Z).flip(
                [-1, -2, -3])
            grad_ys[:, 3] = grad_y.transpose(dim0=-1, dim1=-2)[:, :, :, 1::step_size, 1::step_size].contiguous().view(B,
                                                                                                                      C,
                                                                                                                      X, spl_Y, spl_Z).flip(
                [-1, -2, -3])

        return grad_ys, None, None, None  # [B, K, C, X, Y/s, Z/s]


class EfficientRouteSelector(nn.Module):
    def __init__(self, route_type: str='spatial', step_size: int=2, bidirectional: bool=True, merge_mean: bool=True):
        '''
        select_type: 扫描和融合的方式， 'e' 'efficient'跳跃;  's' 'sequential'连续
        route_type: 选择扫描路径  's' 'spatial';      't' 'temporal';   'st' 'spatiotemporal';   'sy' 'synthetic'
        step_size: 间隔长度，当 type=='efficient'时生效
        bidirectional: 是否反向
        '''
        super(EfficientRouteSelector, self).__init__()

        self.step_size = step_size
        self.bidirectional = bidirectional
        self.merge_mean = merge_mean

        assert int(step_size) > 1, '间隔步骤设置错误'

        if route_type in ['sy', 'synthetic']:
            self.scan = self._scan_synthetic
            self.merge = self._merge_synthetic
            self.K = 12 if bidirectional else 6

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

        self.CSF, self.CMF = EfficientScanFn, EfficientMergeFn
        self.ori_shape, self.spl_shape = None, None



    ### ========== Spatial =======================================================
    def _scan_spatial(self, x: torch.Tensor):
        '''
        Route 1 (Spatial): H*W面，先扫描一行中的每列（W轴）,再逐行扫描图片（H轴），再逐帧堆叠图片（T轴）
        Route 2 (Spatial): H*W面，先扫描一列中的每行（H轴）,再逐列扫描图片（W轴），再逐帧堆叠图片（T轴）
        [B, C, T, H, W] -> [B, K, C, T, H/s, W/s] -> [B, K, C, T * H/s * W/s]
        '''
        self.ori_shape = x.shape    # [B, C, T, H, W]
        xs = self.CSF.apply(x, self.step_size, self.bidirectional)
        self.spl_shape = [B, K, C, T, spl_H, spl_W] = xs.shape  #  [B, K, C, T, H/s, W/s]
        xs = xs.view(B, K, C, T * spl_H * spl_W)  # [B, K, C, T * H/s * W/s]
        return xs


    def _merge_spatial(self, ys: torch.Tensor):
        '''
        [B, K, C, T * H/s * W/s] -> [B, K, C, T, H/s, W/s] ->  [B, C, T, H, W]
        '''
        ys = ys.view(self.spl_shape)   # [B, K, C, T, H/s, W/s]
        y = self.CMF.apply(ys, self.ori_shape, self.step_size, self.bidirectional)  #  [B, C, T, H, W]
        # 跳跃扫描不需要平均
        return y

    ### ========== Temporal =======================================================
    def _scan_temporal(self, x: torch.Tensor):
        '''
        Route 3 (Temporal) :       T*W面，先逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴） 再逐行扫描图片（H轴）
        Route 5 (Spatiotemporal) : T*W面，先再扫描一行中的每列（W轴），逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）
        [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T/s, W/s] -> [B, K, C, H * T/s * W/s]
        '''

        x = x.transpose(dim0=-2, dim1=-3)  # [B, C, T, H, W] ->  [B, C, H, T, W]
        self.ori_shape = x.shape  # [B, C, H, T, W]
        # print('='*100 + '\ntransposed x: \n', x)
        # print(x.shape)

        xs = self.CSF.apply(x, self.step_size, self.bidirectional)
        self.spl_shape = [B, K, C, H, spl_T, spl_W] = xs.shape  # [B, K, C, H, T/s, W/s]
        xs = xs.view(B, K, C, H * spl_T * spl_W)  #[B, K, C, H * T/s * W/s]
        return xs

    def _merge_temporal(self, ys: torch.Tensor):
        '''
        [B, K, C, H * T/s * W/s] -> [B, K, C, H, T/s, W/s] ->  [B, C, H, T, W] -> [B, C, T, H, W]
        '''
        ys = ys.view(self.spl_shape)  # [B, K, C, T, H/s, W/s]
        y = self.CMF.apply(ys, self.ori_shape, self.step_size, self.bidirectional)  # [B, C, H, T, W]
        y = y.transpose(dim0=-2, dim1=-3)  # [B, C, H, T, W] ->  [B, C, T, H, W]
        # 跳跃扫描不需要平均
        return y

    ### ========== Spatiotemporal =======================================================
    def _scan_spatiotemporal(self, x: torch.Tensor):
        '''
        Route 4 (Temporal) : T*H面，先逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）,再扫描一行中的每列（W轴）
        Route 6 (Spatiotemporal) : T*H面，先逐行扫描图片（H轴）,再逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴）
        [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T/s, H/s] -> [B, K, C, W * T/s * H/s]
        '''
        x = x.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W] ->  [B, C, W, T, H]
        self.ori_shape = x.shape  # [B, C, W, T, H]
        # print('='*100 + '\ntransposed x: \n', x)
        # print(x.shape)

        xs = self.CSF.apply(x, self.step_size, self.bidirectional)
        self.spl_shape = [B, K, C, W, spl_T, spl_H] = xs.shape  #  [B, K, C, W, T/s, H/s]
        xs = xs.view(B, K, C, W * spl_T * spl_H)   # [B, K, C, W * T/s * H/s]
        return xs

    def _merge_spatiotemporal(self, ys: torch.Tensor):
        '''
        [B, K, C, W * T/s * H/s] -> [B, K, C, W, T/s, H/s] ->  [B, C, W, T, H] ->  [B, C, T, H, W]
        '''
        ys = ys.view(self.spl_shape)   # [B, K, C, W, T/s, H/s]
        y = self.CMF.apply(ys, self.ori_shape, self.step_size, self.bidirectional)  #   [B, C, W, T, H]
        y = y.permute(0, 1, 3, 4, 2)  # [B, C, T, H, W]
        # 跳跃扫描不需要平均
        return y

    ### ========== Spatiotemporal =======================================================
    def _scan_synthetic(self, x: torch.Tensor):
        '''
        Route 1 (Spatial)  Route 2 (Spatial):
        [B, C, T, H, W] -> [B, K, C, T, H/s, W/s] -> [B, K, C, T * H/s * W/s]

        Route 3 (Temporal) Route 5 (Spatiotemporal) :
        [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T/s, W/s] -> [B, K, C, H * T/s * W/s]

        Route 4 (Temporal) Route 6 (Spatiotemporal) :
        [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T/s, H/s] -> [B, K, C, W * T/s * H/s]

        Return:  [Route 1, Route 2, Route 3, Route 5, Route 4, Route 6]   [B, K', C, H, W, (T * H * W)/(s^2)]
        '''
        self.ori_shape_list, self.spl_shape_list = [], []
        xs_spatial = self._scan_spatial(x)
        self.spl_shape_list.append(self.spl_shape)
        self.ori_shape_list.append(self.ori_shape)

        xs_temporal = self._scan_temporal(x)
        self.spl_shape_list.append(self.spl_shape)
        self.ori_shape_list.append(self.ori_shape)

        xs_spatiotemporal = self._scan_spatiotemporal(x)
        self.spl_shape_list.append(self.spl_shape)
        self.ori_shape_list.append(self.ori_shape)

        return torch.cat([xs_spatial, xs_temporal, xs_spatiotemporal], dim=1)

    def _merge_synthetic(self, ys: torch.Tensor):
        ys_spatial, ys_temporal, ys_spatiotemporal = torch.chunk(ys, 3, dim=1)
        self.spl_shape = self.spl_shape_list[0]
        self.ori_shape = self.ori_shape_list[0]
        y_spatial = self._merge_spatial(ys_spatial)

        self.spl_shape = self.spl_shape_list[1]
        self.ori_shape = self.ori_shape_list[1]
        y_temporal = self._merge_temporal(ys_temporal)

        self.spl_shape = self.spl_shape_list[2]
        self.ori_shape = self.ori_shape_list[2]
        y_spatiotemporal = self._merge_spatiotemporal(ys_spatiotemporal)

        y = torch.stack([y_spatial, y_temporal, y_spatiotemporal], dim=1)

        y = torch.mean(y, dim=1) if self.merge_mean else torch.sum(y, dim=1)
        return y


def test_base_func():
    B, T, H, W, C = 1, 2, 7, 7, 1
    x = torch.zeros([B, T, H, W, C], device=torch.device('cuda:0'))
    x = x.view(B, -1, C)
    for i in range(T * H * W):
        x[:, i, :] = i + 1
    x = x.view(B, C, T, H, W)
    print(x)
    print(x.shape)

    ori_T, ori_H, ori_W = x.shape[-3:]
    step_size = 2
    RS = EfficientRouteSelector(route_type='s', step_size=step_size)
    print(RS.CSF)
    print(RS.CMF)

    SF = RS.CSF()
    xs = SF.forward(x)
    print(xs.flatten(-2))
    print(xs.shape)

    x_grad = SF.backward(xs)[0]
    print(x_grad)
    print(x_grad.shape)

    MF = RS.CMF()
    y = MF.forward(xs, x.shape)
    print(y)
    print(y.shape)

    ys_grad = MF.backward(y)[0]
    print(ys_grad.flatten(-2))
    print(ys_grad.shape)

def test_selector():
    B, T, H, W, C = 1, 6, 4, 4, 1
    x = torch.zeros([B, T, H, W, C], device=torch.device('cuda:0'))
    x = x.view(B, -1, C)
    for i in range(T * H * W):
        x[:, i, :] = i + 1
    x = x.view(B, C, T, H, W)
    print(x)
    print(x.shape)

    step_size = 2

    RS = EfficientRouteSelector(route_type='sy', step_size=step_size)
    xs = RS.scan(x)
    print(xs)

    y = RS.merge(xs)
    print(y)





if __name__ == '__main__':
    ## test
    test_selector()
    ...
