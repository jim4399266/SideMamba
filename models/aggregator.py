import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import complexPyTorch
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

from einops import rearrange
from src.models.module_utils import DyTConv


class Agg_v3:
    def __init_v3__(self, side_dim, patches=12):
        super().__init__()
        self.cls_weight_fc = nn.Sequential(
        nn.Linear(side_dim, side_dim), nn.ReLU(inplace=True),
        nn.Linear(side_dim, 1))
        self.forward = self.forward_v3

    def forward_v3(self, x, flattened=True):
        cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
        cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
        x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)
        return x_cls_n


class Agg_v4:
    def __init_v4__(self, side_dim, patches=12):
        super().__init__()
        self.cls_conv1 = nn.Conv2d(side_dim, side_dim // 2, kernel_size=3, dilation=1, padding=1)  # 局部细节
        self.cls_conv2 = nn.Conv2d(side_dim, side_dim // 2, kernel_size=3, dilation=3, padding=3)  # 大范围上下文

        self.cls_weight_fc = nn.Sequential(
            nn.Linear(side_dim, side_dim), nn.ReLU(inplace=True),
            nn.Linear(side_dim, 1))
        self.forward = self.forward_v4

    def forward_v4(self, x, flattened=True):
        if flattened:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2]))).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # Default not channel first
        conv1 = self.cls_conv1(x)
        conv2 = self.cls_conv2(x)
        x = torch.cat([conv1, conv2], dim=1)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        if flattened:
            x = rearrange(x, 'b h w c -> b (h w) c')
        cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
        cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
        x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)
        return x_cls_n

class Agg_v5:
    def __init_v5__(self, side_dim, patches=12):
        super().__init__()
        self.dyt_conv = DyTConv(side_dim, side_dim, kernel_size=3, dilation=1, padding=1, patches=patches)
        self.cls_weight_fc = nn.Sequential(
            nn.Linear(side_dim, side_dim), nn.ReLU(inplace=True),
            nn.Linear(side_dim, 1))
        self.forward = self.forward_v5

    def forward_v5(self, x, flattened=True):
        if flattened:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2]))).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # Default not channel first
        x = self.dyt_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        if flattened:
            x = rearrange(x, 'b h w c -> b (h w) c')
        cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
        cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
        x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)
        return x_cls_n


class Agg_v6:
    def __init_v6__(self, side_dim, patches=12):
        super().__init__()
        self.dyt_conv = DyTConv(side_dim, 1, kernel_size=3, dilation=1, padding=1, patches=patches)
        # self.cls_weight_fc = nn.Sequential(
        #     nn.Linear(side_dim, side_dim), nn.ReLU(inplace=True),
        #     nn.Linear(side_dim, 1))
        self.forward = self.forward_v6

    def forward_v6(self, x, flattened=True):
        x_s = x
        if flattened:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2]))).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # Default not channel first
        x = self.dyt_conv(x)   # 这里将 channel 维度降到1
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        if flattened:
            x = rearrange(x, 'b h w c -> b (h w) c')

        x_weight = torch.softmax(x, dim=-2)
        x_s = x_s * x_weight
        x_cls_n = torch.sum(x_s, dim=-2, keepdim=True)

        return x_cls_n


class Agg_v7:
    def __init_v7__(self, side_dim, patches=12):
        super().__init__()
        self.cls_conv1 = nn.Conv2d(side_dim, 1, kernel_size=3, dilation=1, padding=1)  # 局部细节
        self.cls_conv2 = nn.Conv2d(side_dim, 1, kernel_size=3, dilation=3, padding=3)  # 大范围上下文

        # self.cls_weight_fc = nn.Sequential(
        #     nn.Linear(side_dim, side_dim), nn.ReLU(inplace=True),
        #     nn.Linear(side_dim, 1))
        self.forward = self.forward_v7

    def forward_v7(self, x, flattened=True):
        x_s = x
        if flattened:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2]))).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # Default not channel first
        conv1 = self.cls_conv1(x)
        conv2 = self.cls_conv2(x)
        x = conv1 + conv2  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        if flattened:
            x = rearrange(x, 'b h w c -> b (h w) c')


        cls_weight = torch.softmax(x, dim=-2).squeeze()  # BT L C  ->  BT L
        # cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
        x_cls_n = torch.einsum('blc,bl->bc', [x_s, cls_weight]).unsqueeze(1)
        return x_cls_n

class Agg_v8:
    def __init_v8__(self, side_dim, patches=12):
        super().__init__()
        self.cls_conv1 = ComplexConv2d(side_dim, 1, kernel_size=3, dilation=1, padding=1)  # 局部细节
        self.cls_conv2 = ComplexConv2d(side_dim, 1, kernel_size=3, dilation=3, padding=3)  # 大范围上下文
        self.forward = self.forward_v8

    def forward_v8(self, x, flattened=True):
        x_s = x
        if flattened:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2]))).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # Default not channel first
        x = torch.complex(x, torch.zeros_like(x))
        conv1 = self.cls_conv1(x)
        conv2 = self.cls_conv2(x)
        x = torch.abs(conv1 + conv2)  # Convert complex responses to real attention logits.
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        if flattened:
            x = rearrange(x, 'b h w c -> b (h w) c')

        cls_weight = torch.softmax(x, dim=-2).squeeze()  # BT L C  ->  BT L
        # cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
        x_cls_n = torch.einsum('blc,bl->bc', [x_s, cls_weight]).unsqueeze(1)
        return x_cls_n


class ClsAggregator(nn.Module, Agg_v3, Agg_v4, Agg_v5, Agg_v6, Agg_v7, Agg_v8):
    def __init__(self, side_dim, patches=14, agg_type=''):
        nn.Module.__init__(self)
        if agg_type == 'v3':
            self.__init_v3__(side_dim, patches)
        elif agg_type == 'v4':
            self.__init_v4__(side_dim, patches)
        elif agg_type == 'v5':
            self.__init_v5__(side_dim, patches)
        elif agg_type == 'v6':
            self.__init_v6__(side_dim, patches)
        elif agg_type == 'v7':
            self.__init_v7__(side_dim, patches)
        elif agg_type == 'v8':
            self.__init_v8__(side_dim, patches)
        else:
            raise NotImplementedError
