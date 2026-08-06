import torch
import torch.nn as nn

from .efficient_scan import EfficientRouteSelector
from .sequential_scan import SequentialRouteSelector

from .csm_triton import Cross3DScanTritonF, Cross3DMergeTritonF, RouteSelectorTriton

def create_route_selector(select_type: str='sequential', route_type: str='spatial', step_size: int=2, bidirectional: bool=True, merge_mean: bool=True, with_triton=True):
    if select_type in ['efficient', 'e']:
        return EfficientRouteSelector(route_type=route_type, step_size=step_size, bidirectional=bidirectional, merge_mean=merge_mean)
    elif select_type in ['spatial', 's']:
        if with_triton:
            return RouteSelectorTriton(route_type=route_type, bidirectional=bidirectional, merge_mean=merge_mean)
        else:
            raise NotImplementedError
            # return SequentialRouteSelector(route_type=route_type, bidirectional=bidirectional, merge_mean=merge_mean)
    else:
        raise NotImplementedError


# class RouteSelector(nn.Module):
#     def __init__(self, select_type: str='sequential', route_type: str='spatial', bidirectional: bool=True, merge_mean: bool=True, with_triton=True):
#         '''
#         select_type: 扫描方式    's' 'sequential' 连续扫描;   'e' 'efficient'  跳跃扫描
#         route_type: 选择扫描路径  's' 'spatial';      't' 'temporal';   'st' 'spatiotemporal';   'sy' 'synthetic'
#         step_size: 间隔长度，当 type=='efficient'时生效
#         bidirectional: 是否反向
#         '''
#         super(RouteSelector, self).__init__()
#
#         self.bidirectional = bidirectional
#         self.merge_mean = merge_mean
#
#         if route_type in ['sy', 'synthetic']:
#             self.scan = self._scan_synthetic
#             self.merge = self._merge_synthetic
#             self.K = 12 if bidirectional else 6
#
#         elif route_type in ['s', 'spatial']:
#             self.scan = self._scan_spatial
#             self.merge = self._merge_spatial
#             self.K = 4 if bidirectional else 2
#
#         elif route_type in ['t', 'temporal']:
#             self.scan = self._scan_temporal
#             self.merge = self._merge_temporal
#             self.K = 4 if bidirectional else 2
#
#         elif route_type in ['st', 'spatiotemporal']:
#             self.scan = self._scan_spatiotemporal
#             self.merge = self._merge_spatiotemporal
#             self.K = 4 if bidirectional else 2
#         else:
#             raise NotImplementedError(f'route_type {route_type} not implemented')
#
#         if with_triton:
#             if select_type in ['s', 'sequential']:
#                 self.CSF = Cross3DScanTritonF
#                 self.CMF = Cross3DMergeTritonF
#             elif select_type in ['e', 'efficient']:
#                 self.CSF = EfficientCross3DScanTritonF
#                 ...
#             else:
#                 raise NotImplementedError
#         else:
#             raise NotImplementedError
#
#     ### ========== Spatial =======================================================
#     def _scan_spatial(self, x: torch.Tensor):
#         '''
#         Route 1 (Spatial): H*W面，先扫描一行中的每列（W轴）,再逐行扫描图片（H轴），再逐帧堆叠图片（T轴）
#         Route 2 (Spatial): H*W面，先扫描一列中的每行（H轴）,再逐列扫描图片（W轴），再逐帧堆叠图片（T轴）
#         [B, C, T, H, W] -> [B, K, C, T, H, W] -> [B, K, C, T * H * W]
#         '''
#         # K = 4 if self.bidirectional else 2
#         self.ori_shape = x.shape    # [B, C, T, H, W]
#         with torch.cuda.device(x.device):
#             y = self.CSF.apply(x, False, 0, self.bidirectional)
#         self.tar_shape = [B, K, C, T, H, W] = y.shape  # [B, K, C, T, H, W]
#         y = y.view(B, K, C, T * H * W)   # [B, K, C, T * H * W]
#         return y
#
#
#     def _merge_spatial(self, y: torch.Tensor):
#         '''
#         [B, K, C, T * H * W] -> [B, K, C, T, H, W] ->  [B, C, T, H, W]
#         '''
#         y = y.view(self.tar_shape)   # [B, K, C, T, H, W]
#         with torch.cuda.device(y.device):
#             x = self.CMF.apply(y, False, 0, self.bidirectional)
#         if self.merge_mean:
#             K = y.shape[1]
#             x = x / float(K)
#         return x  #  [B, C, T, H, W]
#
#     ### ========== Temporal =======================================================
#     def _scan_temporal(self, x: torch.Tensor):
#         '''
#         Route 3 (Temporal) :       T*W面，先逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴） 再逐行扫描图片（H轴）
#         Route 5 (Spatiotemporal) : T*W面，先再扫描一行中的每列（W轴），逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）
#         [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T, W] -> [B, K, C, H * T * W]
#         '''
#         # K = 4 if self.bidirectional else 2
#         x = x.transpose(dim0=-2, dim1=-3)  # [B, C, T, H, W] ->  [B, C, H, T, W]
#         self.ori_shape = x.shape  # [B, C, H, T, W]
#         with torch.cuda.device(x.device):
#             y = self.CSF.apply(x, False, 0, self.bidirectional)
#         self.tar_shape = [B, K, C, H, T, W] = y.shape  # [B, K, C, H, T, W]
#         y = y.view(B, K, C, H * T * W)  # [B, K, C, H * T * W]
#         return y
#
#     def _merge_temporal(self, y: torch.Tensor):
#         '''
#         [B, K, C, H * T * W] -> [B, K, C, H, T, W] ->  [B, C, H, T, W] -> [B, C, T, H, W]
#         '''
#         y = y.view(self.tar_shape)  # [B, K, C, T, H, W]
#         with torch.cuda.device(y.device):
#             x = self.CMF.apply(y, False, 0, self.bidirectional)  # [B, C, H, T, W]
#         x = x.transpose(dim0=-2, dim1=-3)  # [B, C, H, T, W] ->  [B, C, T, H, W]
#         if self.merge_mean:
#             K = y.shape[1]
#             x = x / float(K)
#         return x
#
#     ### ========== Spatiotemporal =======================================================
#     def _scan_spatiotemporal(self, x: torch.Tensor):
#         '''
#         Route 4 (Temporal) : T*H面，先逐帧扫描每个像素点（T轴）,再逐行扫描图片（H轴）,再扫描一行中的每列（W轴）
#         Route 6 (Spatiotemporal) : T*H面，先逐行扫描图片（H轴）,再逐帧扫描每个像素点（T轴）,再扫描一行中的每列（W轴）
#         [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T, H] -> [B, K, C, W * T * H]
#         '''
#         # K = 4 if self.bidirectional else 2
#         x = x.permute(0, 1, 4, 2, 3)  # [B, C, T, H, W] ->  [B, C, W, T, H]
#         self.ori_shape = x.shape  # [B, C, W, T, H]
#
#         with torch.cuda.device(x.device):
#             y = self.CSF.apply(x, False, 0, self.bidirectional)
#         self.tar_shape = [B, K, C, W, T, H] = y.shape  # [B, K, C, W, T, H]
#         y = y.view(B, K, C, W * T * H) # [B, K, C, W * T * H]
#         return y
#
#     def _merge_spatiotemporal(self, y: torch.Tensor):
#         '''
#         [B, K, C, W * T * H] -> [B, K, C, W, T, H] ->  [B, C, W, T, H] ->  [B, C, T, H, W]
#         '''
#         y = y.view(self.tar_shape)   # [B, K, C, W, T/s, H/s]
#         with torch.cuda.device(y.device):
#             x = self.CMF.apply(y, False, 0, self.bidirectional) #   [B, C, W, T, H]
#         x = x.permute(0, 1, 3, 4, 2)  # [B, C, T, H, W]
#         if self.merge_mean:
#             K = y.shape[1]
#             x = x / float(K)
#         return x
#
#     ### ========== Spatiotemporal =======================================================
#     def _scan_synthetic(self, x: torch.Tensor):
#         '''
#         Route 1 (Spatial)  Route 2 (Spatial):
#         [B, C, T, H, W] -> [B, K, C, T, H, W] -> [B, K, C, T * H * W]
#
#         Route 3 (Temporal) Route 5 (Spatiotemporal) :
#         [B, C, T, H, W] -> [B, C, H, T, W] -> [B, K, C, H, T, W] -> [B, K, C, H * T * W]
#
#         Route 4 (Temporal) Route 6 (Spatiotemporal) :
#         [B, C, T, H, W] -> [B, C, W, T, H] -> [B, K, C, W, T, H] -> [B, K, C, W * T * H]
#
#         Return:  [Route 1, Route 2, Route 3, Route 5, Route 4, Route 6]   [B, K', C, H, W, T * H * W]
#         '''
#         self.tar_shape_list = []
#         y_spatial = self._scan_spatial(x)
#         self.tar_shape_list.append(self.tar_shape)
#
#         y_temporal = self._scan_temporal(x)
#         self.tar_shape_list.append(self.tar_shape)
#
#         y_spatiotemporal = self._scan_spatiotemporal(x)
#         self.tar_shape_list.append(self.tar_shape)
#         y = torch.cat([y_spatial, y_temporal, y_spatiotemporal], dim=1)
#         return y
#
#     def _merge_synthetic(self, ys: torch.Tensor):
#         ys_spatial, ys_temporal, ys_spatiotemporal = torch.chunk(ys, 3, dim=1)
#         self.tar_shape = self.tar_shape_list[0]
#         x_spatial = self._merge_spatial(ys_spatial)
#
#         self.tar_shape = self.tar_shape_list[1]
#         x_temporal = self._merge_temporal(ys_temporal)
#
#         self.tar_shape = self.tar_shape_list[2]
#         x_spatiotemporal = self._merge_spatiotemporal(ys_spatiotemporal)
#
#
#         x = torch.stack([x_spatial, x_temporal, x_spatiotemporal], dim=1)
#
#         x = torch.mean(x, dim=1) if self.merge_mean else torch.sum(x, dim=1)
#         return x


# if __name__ == '__main__':
#     ## test
#     B, T, H, W, C = 1, 2, 4, 4, 1
#     x = torch.zeros([B, T, H, W, C])
#     x = x.view(B, -1, C)
#     for i in range(T * H * W):
#         x[:, i, :] = i + 1
#     x = x.view(B, C, T, H, W)
#     print(x)
#     print(x.shape)
#
#     ori_T, ori_H, ori_W = x.shape[-3:]
#     step_size = 2
#     RSF = RouteSelectorTriton(select_type='s', route_type='sy')
#     print(RSF.ScanFn)
#     print(RSF.MergeFn)
#
#     xs = RSF.ScanFn().forward(x)
#     print(xs.flatten(-2))
#     print(xs.shape)
#
#     x_grad = RSF.ScanFn().backward(xs)[0]
#     print(x_grad)
#     print(x_grad.shape)