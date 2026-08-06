from typing import Any
import torch
import warnings
import math
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from typing import Union, List, Tuple

###  ====================================== BaseFn =================================================
###  ===========================================================================================================
class SequentialScanFn(torch.autograd.Function):
    # 输入维度是： [B, C, X, Y, Z] ，对 Y , Z 维度进行扫描， 在 X 维度进行堆叠，就不说 T, H, W了，不考虑填充的情况
    @staticmethod
    def forward(ctx, x: torch.Tensor, merge_mean: bool=True, bidirectional: bool=True):
        # [B, C, X, Y, Z] -> [B, K, C, X, Y, Z]
        # x 为待扫描的张量，已完成padding（如需）
        B, C, X, Y, Z = x.shape
        K = 4 if bidirectional else 2

        ctx.ori_shape = [B, C, X, Y, Z]
        ctx.bidirectional = bidirectional
        ctx.merge_mean = merge_mean
        ctx.K = K

        xs = x.new_empty((B, K, C, X, Y, Z))
        xs[:, 0] = x.contiguous().view(B, C, X, Y, Z)
        xs[:, 1] = x.transpose(dim0=-1, dim1=-2).contiguous().view(B, C, X, Y, Z)
        if bidirectional:
            xs[:, 2] = x.contiguous().view(B, C, X, Y, Z).flip([-1, -2, -3])
            xs[:, 3] = x.transpose(dim0=-1, dim1=-2).contiguous().view(B, C, X, Y, Z).flip([-1, -2, -3])

        return xs  #  [B, K, C, X, Y, Z]

    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor): #  [B, K, C, X, Y, Z]  -> [B, C, X, Y, Z]
        B, C, X, Y, Z = ctx.ori_shape
        K = ctx.K
        bidirectional = ctx.bidirectional
        merge_mean = ctx.merge_mean
        grad_x = grad_xs.new_empty((B, K, C, X, Y, Z))

        grad_x[:, 0] = grad_xs[:, 0].reshape(B, C, X, Y, Z)
        grad_x[:, 1] = grad_xs[:, 1].reshape(B, C, X, Z, Y).transpose(dim0=-1, dim1=-2)
        if bidirectional:
            grad_x[:, 2] = grad_xs[:, 2].reshape(B, C, X, Y, Z).flip([-1, -2, -3])
            grad_x[:, 3] = grad_xs[:, 3].reshape(B, C, X, Z, Y).transpose( dim0=-1, dim1=-2).flip([-1, -2, -3])
        grad_x = torch.mean(grad_x, dim=1) if merge_mean else torch.sum(grad_x, dim=1)
        return grad_x, None, None  # [B, C, X, Y, Z]


class SequentialMergeFn(torch.autograd.Function):
    # [B, K, C, X, Y, Z]
    @staticmethod
    def forward(ctx, ys: torch.Tensor, merge_mean: bool=True, bidirectional: bool=True): # [B, K, C, X, Y, Z]  -> [B, C, X, Y, Z]
        B, K, C, X, Y, Z = ys.shape

        ctx.target_shape = [B, K, C, X, Y, Z]

        ctx.merge_mean = merge_mean
        ctx.bidirectional = bidirectional

        y = ys.new_empty((B, K, C, X, Y, Z))

        y[:, 0] = ys[:, 0].reshape(B, C, X, Y, Z)
        y[:, 1] = ys[:, 1].reshape(B, C, X, Z, Y).transpose(dim0=-1, dim1=-2)
        if bidirectional:
            y[:, 2] = ys[:, 2].reshape(B, C, X, Y, Z).flip([-1, -2, -3])
            y[:, 3] = ys[:, 3].reshape(B, C, X, Z, Y).transpose(dim0=-1, dim1=-2).flip([-1, -2, -3])
        y = torch.mean(y, dim=1) if merge_mean else torch.sum(y, dim=1)
        return y  # [B, C, X, Y, Z]

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # [B, C, X, Y, Z] -> [B, K, C, X, Y, Z]
        B, K, C, X, Y, Z = ctx.target_shape
        bidirectional = ctx.bidirectional

        grad_ys = grad_y.new_empty((B, K, C, X, Y, Z))
        grad_ys[:, 0] = grad_y.contiguous().view(B, C, X, Y, Z)
        grad_ys[:, 1] = grad_y.transpose(dim0=-1, dim1=-2).contiguous().view(B, C, X, Y, Z)
        if bidirectional:
            grad_ys[:, 2] = grad_y.contiguous().view(B, C, X, Y, Z).flip([-1, -2, -3])
            grad_ys[:, 3] = grad_y.transpose(dim0=-1, dim1=-2).contiguous().view(B, C, X, Y, Z).flip([-1, -2, -3])

        return grad_ys, None, None  # [B, K, C, X, Y, Z]


class SequentialRouteSelector(nn.Module):
    def __init__(self, route_type: str='spatial', bidirectional: bool=True, merge_mean: bool=True):
        '''
        route_type: 选择扫描路径  's' 'spatial';      't' 'temporal';   'st' 'spatiotemporal';   'sy' 'synthetic'
        step_size: 间隔长度，当 type=='efficient'时生效
        bidirectional: 是否反向
        '''
        super(SequentialRouteSelector, self).__init__()

        self.bidirectional = bidirectional
        self.merge_mean = merge_mean

        self.ScanFn, self.MergeFn = SequentialScanFn, SequentialMergeFn
        self.scan, self.merge = None, None


        if route_type in ['s', 'spatial']:
            self.scan = self._scan_spatial
            self.merge = self._merge_spatial
        elif route_type in ['t', 'temporal']:
            self.scan = self._scan_temporal
            self.merge = self._merge_temporal
        elif route_type in ['st', 'spatiotemporal']:
            self.scan = self._scan_spatiotemporal
            self.merge = self._merge_spatiotemporal
        elif route_type in ['sy', 'synthetic']:
            self.scan = self._scan_synthetic
            self.merge = self._merge_synthetic
        else:
            raise NotImplementedError(f'route_type {route_type} not implemented')



    ### ========== Spatial =======================================================
    def _scan_spatial(self, x: torch.Tensor):
        '''
        Route 1 (Spatial): H*W面，先扫描一行中的每列（W轴）,再逐行扫描图片（H轴），再逐帧堆叠图片（T轴）
        Route 2 (Spatial): H*W面，先扫描一列中的每行（H轴）,再逐列扫描图片（W轴），再逐帧堆叠图片（T轴）
        [B, C, T, H, W] -> [B, K, C, T, H, W] -> [B, K, C, T * H * W]
        '''
        self.ori_shape = x.shape    # [B, C, T, H, W]
        xs = self.ScanFn.apply(x, self.merge_mean, self.bidirectional)
        self.tar_shape = [B, K, C, T, H, W] = xs.shape  #  [B, K, C, T, H, W]

        return xs.view(B, K, C, T * H * W)   # [B, K, C, T * H * W]


    def _merge_spatial(self, ys: torch.Tensor):
        '''
        [B, K, C, T * H * W] -> [B, K, C, T, H, W] ->  [B, C, T, H, W]
        '''
        ys = ys.view(self.tar_shape)   # [B, K, C, T, H, W]
        y = self.MergeFn.apply(ys, self.merge_mean, self.bidirectional)  #  [B, C, T, H, W]

        return y

    ### ========== Temporal =======================================================
    def _scan_temporal(self, x: torch.Tensor):
        '''
        Route 3 (Temporal) :       T*W面，先逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴） 再逐行扫描图片（H轴）
        Route 5 (Spatiotemporal) : T*W面，先再扫描一行中的每列（W轴），逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）
        [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T, W] -> [B, K, C, H * T * W]
        '''
        self.ori_shape = x.shape  # [B, C, T, H, W]
        x = x.transpose(dim0=-2, dim1=-3)  # [B, C, T, H, W] ->  [B, C, H, T, W]
        # print('transposed x: \n', x)
        # print(x.shape)


        xs = self.ScanFn.apply(x, self.merge_mean, self.bidirectional)
        self.tar_shape = [B, K, C, H, T, W] = xs.shape  # [B, K, C, H, T, W]

        return xs.view(B, K, C, H * T * W)  #[B, K, C, H * T * W]

    def _merge_temporal(self, ys: torch.Tensor):
        '''
        [B, K, C, H * T * W] -> [B, K, C, H, T, W] ->  [B, C, H, T, W] -> [B, C, T, H, W]
        '''
        ys = ys.view(self.tar_shape)  # [B, K, C, T, H, W]
        y = self.MergeFn.apply(ys, self.merge_mean, self.bidirectional)  # [B, C, H, T, W]
        y = y.transpose(dim0=-2, dim1=-3)  # [B, C, H, T, W] ->  [B, C, T, H, W]
        return y

    ### ========== Spatiotemporal =======================================================
    def _scan_spatiotemporal(self, x: torch.Tensor):
        '''
        Route 4 (Temporal) : T*H面，先逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）,再扫描一行中的每列（W轴）
        Route 6 (Spatiotemporal) : T*H面，先逐行扫描图片（H轴）,再逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴）
        [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T, H] -> [B, K, C, W * T * H]
        '''
        self.ori_shape = x.shape    # [B, C, T, H, W]
        x = x.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W] ->  [B, C, W, T, H]
        # print('transposed x: \n', x)
        # print(x.shape)

        xs = self.ScanFn.apply(x, self.merge_mean, self.bidirectional)
        self.tar_shape = [B, K, C, W, T, H] = xs.shape  #  [B, K, C, W, T, H]

        return xs.view(B, K, C, W * T * H)   # [B, K, C, W * T * H]

    def _merge_spatiotemporal(self, ys: torch.Tensor):
        '''
        [B, K, C, W * T * H] -> [B, K, C, W, T, H] ->  [B, C, W, T, H] ->  [B, C, T, H, W]
        '''
        ys = ys.view(self.tar_shape)   # [B, K, C, W, T/s, H/s]
        y = self.MergeFn.apply(ys, self.merge_mean, self.bidirectional)  #   [B, C, W, T, H]
        y = y.permute(0, 1, 3, 4, 2)  # [B, C, T, H, W]
        return y

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
        xs_spatial = self._scan_spatial(x)
        self.tar_shape_list.append(self.tar_shape)

        xs_temporal = self._scan_temporal(x)
        self.tar_shape_list.append(self.tar_shape)

        xs_spatiotemporal = self._scan_spatiotemporal(x)
        self.tar_shape_list.append(self.tar_shape)
        return torch.cat([xs_spatial, xs_temporal, xs_spatiotemporal], dim=1)

    def _merge_synthetic(self, ys: torch.Tensor):
        ys_spatial, ys_temporal, ys_spatiotemporal = torch.chunk(ys, 3, dim=1)
        self.tar_shape = self.tar_shape_list[0]
        y_spatial = self._merge_spatial(ys_spatial)

        self.tar_shape = self.tar_shape_list[1]
        y_temporal = self._merge_temporal(ys_temporal)

        self.tar_shape = self.tar_shape_list[2]
        y_spatiotemporal = self._merge_spatiotemporal(ys_spatiotemporal)

        y = torch.stack([y_spatial, y_temporal, y_spatiotemporal], dim=1)

        y = torch.mean(y, dim=1) if self.merge_mean else torch.sum(y, dim=1)
        return y





if __name__ == '__main__':
    ## test
    B, T, H, W, C = 1, 2, 4, 4, 1
    x = torch.zeros([B, T, H, W, C])
    x = x.view(B, -1, C)
    for i in range(T * H * W):
        x[:, i, :] = i + 1
    x = x.view(B, C, T, H, W)
    print(x)
    print(x.shape)

    ori_T, ori_H, ori_W = x.shape[-3:]
    step_size = 2
    RSF = RouteSelector(select_type='e', route_type='s', step_size=step_size)
    print(RSF.ScanFn)
    print(RSF.MergeFn)

    xs = RSF.ScanFn().forward(x)
    print(xs.flatten(-2))
    print(xs.shape)

    x_grad = RSF.ScanFn().backward(xs)[0]
    print(x_grad)
    print(x_grad.shape)