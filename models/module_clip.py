"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union
from pathlib import Path
import hashlib
import os
import urllib
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from random import sample
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath

from src.models.module_utils import EncoderOutput, LayerNorm, QuickGELU

from src.models.module_vmamba import SideVMamba


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
}

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def conv_3x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


def download_model(pretrained_clip_name):
    root = './clip_pretrain'
    # root = '/home/hpluo/lhp/Codes/PycharmCodes/Video-Text-Retrieval/CLIP-models'
    model_path = os.path.join(root, _PT_NAME[pretrained_clip_name])
    return model_path

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1).contiguous()  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x




class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttnCBlock(nn.Module):
    def __init__(self, dim, side_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=5, T=12):
        super().__init__()
        self.T = T
        self.norm1 = bn_3d(dim)
        self.pw_conv1 = conv_1x1x1(dim, side_dim, 1)
        self.pw_conv2 = conv_1x1x1(side_dim, dim, 1)
        if kernel_size == 5:
            self.dw_conv1 = conv_5x3x3(side_dim, side_dim, groups=side_dim)
        elif kernel_size == 7:
            self.dw_conv1 = conv_7x5x5(side_dim, side_dim, groups=side_dim)
        elif kernel_size == 1:
            self.dw_conv1 = conv_3x1x1(side_dim, side_dim, groups=side_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim* mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = nn.MultiheadAttention(dim, dim//64, dropout=0.)
        self.ln_1 = LayerNorm(dim)

        side_attn_std = dim ** -0.5
        side_fc_std = (2 * dim) ** -0.5
        side_proj_std = (dim ** -0.5) * ((2 * 12) ** -0.5)
        for name, p in self.named_parameters():
            if 'mlp.fc1.weight' in name:
                nn.init.normal_(p, std=side_fc_std)
            elif 'mlp.fc2.weight' in name:
                nn.init.normal_(p, std=side_proj_std)
            elif 'pw_conv1.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'pw_conv2.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'dw_conv1.weight' in name:
                nn.init.normal_(p, std=side_attn_std)
            elif 'attn.in_proj_weight' in name:
                nn.init.normal_(p, std = side_attn_std)
            elif 'attn.out_proj.weight' in name:
                nn.init.normal_(p, std = side_proj_std)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.BatchNorm3d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = None # self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def shift_token(self, x_token): # [1, bt, dilation]
        c = x_token.shape[-1]
        fold = c // 2
        x_token = rearrange(x_token, 'n (b t) dilation -> n b t dilation', t=self.T)
        out = torch.zeros_like(x_token)
        out[:, :, :-1, :fold] = x_token[:, :, 1:, :fold]
        out[:, :, 1:, fold:] = x_token[:, :, :-1, fold:]
        out = rearrange(out, 'n b t dilation -> n (b t) dilation')
        return out

    def forward(self, x, x_token=None, side_position_embeddings=None, layer_id=None):
        n, bt, d = x.size()
        h = int(x.shape[0] ** 0.5)
        x = rearrange(x, '(h w) (b t) dilation -> b dilation t h w', h=h, t=self.T)
        x = x + self.drop_path(self.pw_conv2(self.dw_conv1(self.pw_conv1(self.norm1(x)))))
        x = rearrange(x, 'b dilation t h w -> (h w) (b t) dilation', h=h, t=self.T)

        ## shift class token
        x_token = self.shift_token(x_token)
        xt = torch.cat([x_token, x], dim=0)
        xt = xt.permute(1, 0, 2).contiguous()
        xt = xt + side_position_embeddings.to(x.dtype)
        xt = xt.permute(1, 0, 2).contiguous()
        xt = self.attention(self.ln_1(xt))
        x = x + xt[1:, :, :]

        x_ = x
        x = rearrange(x, '(h w) (b t) dilation -> b dilation t h w', h=h, t=self.T)
        x = self.norm2(x)
        x = rearrange(x, 'b dilation t h w -> (h w) (b t) dilation', h=h, t=self.T)
        x = x_ + self.drop_path(self.mlp(x))
        return x
  

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1.float()(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1.float()(x)))

        # FFN
        if use_checkpoint:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2.float()(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2.float()(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
        all_hidden_states = () if output_hidden_states else None
        for block in self.resblocks:
            x = block(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        return EncoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
        )



class VisualTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512, linear_patch='2d', T=12, side_network: Union[SideVMamba]=None):
        super().__init__()
        self.img_size = img_size
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.T = T
        self.layers = layers
        self.transformer = Transformer(width, layers, heads)

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


        self.side_mamba_v = side_network
        if side_network is not None:
            self.side_layer_index, self.side_layer_route_index = self.side_mamba_v.set_side_layer_index(self.layers)
            side_scale = self.side_mamba_v.side_dim ** -0.5 # (output_dim+self.side_dim) ** -0.5
            self.side_proj = nn.Parameter(side_scale * torch.zeros(self.side_mamba_v.side_dim, width))


    def forward_core_o(self, x: torch.Tensor, video_frame=-1, return_hidden=False):
        '''
        旧方法，将所有ViT输出打包输入SiM
        '''

        image = x
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2).contiguous()  # BT N D -> N BT D
        all_hidden = self.transformer(x, output_hidden_states=True).hidden_states  # L N BT D
        # trans_output = self.transformer(x, output_hidden_states=True)
        # last_hidden, all_hidden = trans_output.last_hidden_state, trans_output.hidden_states
        # 检查所有张量形状是否相同
        shapes = [t.shape for t in all_hidden]
        assert all(s == shapes[0] for s in shapes), "张量形状不一致！"
        all_hidden = torch.stack(all_hidden).permute(0, 2, 1, 3).contiguous()  # L BT N D
        # all_hidden = rearrange(torch.stack(all_hidden), 'l n (b t) dilation -> l b t n dilation', t=self.T).contiguous()  # L B T N D
        last_hidden = rearrange(all_hidden[-1], '(b t) l dilation -> b t l dilation', t=self.T)
        # x_trans_cls = rearrange(last_hidden[..., 0, :], '(b t) dilation -> b t dilation', t=self.T)  # BT D -> B T D

        side_output = self.side_mamba_v(image, all_hidden)


        ## 已将 cls 放到 x_side_hidden 的第 0 位
        x_side_hidden = side_output.last_hidden_state # B T N D
        x_side_hidden = x_side_hidden @ self.side_proj

        # x_hidden = last_hidden + self.side_ratio * x_side_hidden  # side_ratio_1
        # x_hidden = (1 - self.side_ratio) * last_hidden + self.side_ratio * x_side_hidden # side_ratio_2

        x_hidden = last_hidden + x_side_hidden

        # Move the three lines below to `encode_image` for entire hidden sequence
        x_hidden = self.ln_post(x_hidden) @ self.proj
        x_cls = x_hidden[..., 0, :]
        if return_hidden:
            return EncoderOutput(
                pooler_output=x_cls,
                last_hidden_state=x_hidden,
            )
        else:
            return EncoderOutput(
                pooler_output=x_cls,
                # last_hidden_state=x_hidden,
            )

    def forward_core_f(self, x: torch.Tensor, video_frame=-1, return_hidden=False, return_all_hidden=False):
        '''
        测试不走Side的显存占用
        '''
        # ======================================  Backbone 输入预处理 =============================================================
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # ======================================  分层计算  =============================================================

        x = x.permute(1, 0, 2).contiguous()  # BT N D -> N BT D

        for i in range(self.layers):
            x = self.transformer.resblocks[i](x)

        # x = x.permute(1, 0, 2).contiguous()  # N BT D -> BT N D
        x = rearrange(x, 'l (b t) d -> b t l d', t=self.T) # N BT D -> B T N D


        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x) @ self.proj
        x_cls = x[..., 0, :]
        return EncoderOutput(
            pooler_output=x_cls,
            last_hidden_state=x,
        )


    def forward_core_n(self, x: torch.Tensor, video_frame=-1, return_hidden=False, return_all_hidden=False):
        '''
        新方法，将SiM的forward搬出来，每层ViT直接输入SiM
        '''
        all_hidden_states = () if return_hidden else None

        # ======================================  Side 输入预处理 =============================================================
        # x_side = self.side_mamba_v.patch_embed(rearrange(x, '(b t) c h w -> b c t h w', t=self.T), channel_first=True)
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)
        x_side = self.side_mamba_v.side_pre_bn3d(self.side_mamba_v.side_conv1(x_side))
        x_side = rearrange(x_side,'b c t h w -> (b t) (h w) c').contiguous()


        BT, L, C = x_side.shape  # BT L C

        ## 为 side mamba 添加cls token， 并将vision transformer 中的cls token 调整到与 mamba 相同的位置
        if self.side_mamba_v.if_cls_token:
            cls_token = self.side_mamba_v.cls_token.expand(BT, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            self.side_mamba_v.token_position = 0
            x_side = torch.cat((cls_token, x_side), dim=1)
            L = x_side.shape[1]

        ## pos embedding
        if self.side_mamba_v.if_abs_pos_embed:
            x_side = self.side_mamba_v.PE.add_pos_embed(x_side)


        # ======================================  Backbone 输入预处理 =============================================================
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # ======================================  分层计算  =============================================================

        x = x.permute(1, 0, 2).contiguous()  # BT N D -> N BT D

        for i in range(self.layers):
            x = self.transformer.resblocks[i](x)
            if i in self.side_layer_index:
                j = self.side_layer_index.index(i)

                # 将 x_side 和 x_trans 进行拼接
                xs2xt = x.permute(1, 0, 2).contiguous()
                xs2xt = self.side_mamba_v.side_linears[j](self.side_mamba_v.side_lns[j](xs2xt))

                # Original fixed Side/CLIP fusion:
                x_side = x_side * 0.5 + xs2xt * 0.5

                x_cls = x_side[..., 0, :].unsqueeze(1)  # BT 1 C
                x_hidden = x_side[..., 1:, :]

                layer = self.side_mamba_v.blocks[j]

                if self.side_mamba_v.cls_interaction == 'v1':
                    ### v1  cls不参与block的计算
                    x_hidden = layer(x_hidden, flattened=True)

                elif self.side_mamba_v.cls_interaction == 'v2':
                    ### v2  将CLS Token广播到每个空间位置，并与图像特征相加，注入全局信息，再通过全局池化提取CLS信息
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    x_cls_n = torch.mean(x_hidden, dim=1, keepdim=True)
                    x_cls = (x_cls + x_cls_n) / 2.0


                elif self.side_mamba_v.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
                    ### v3  将CLS Token广播到每个空间位置，再通过空间注意力聚合获得CLS

                    ### v4  将CLS Token广播到每个空间位置，再通过多尺度卷积融合获得CLS
                    ### 使用不同膨胀率（dilation）的卷积核，捕获不同尺度的上下文信息。
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    # x_cls_n = self.cls_agg(x, flattened=True)
                    x_cls_n = self.side_mamba_v.cls_agg[j](x_hidden, flattened=True)

                    x_cls = (x_cls + x_cls_n) / 2.0

                x_side = torch.cat([x_cls, x_hidden], dim=-2)
                if return_all_hidden:
                    all_hidden_states = all_hidden_states + (x_side,)


        # x = x.permute(1, 0, 2).contiguous()  # N BT D -> BT N D
        x = rearrange(x, 'l (b t) d -> b t l d', t=self.T) # N BT D -> B T N D
        x_side = rearrange(x_side, '(b t) l d -> b t l d', t=self.T)

        x_side = self.side_mamba_v.side_post_ln(x_side)

        ## 已将 cls 放到 x_side 的第 0 位
        side_output = x_side @ self.side_proj
        x = x + side_output

        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x) @ self.proj
        x_cls = x[..., 0, :]
        return EncoderOutput(
            pooler_output=x_cls,
            last_hidden_state=x if return_hidden else None,
            hidden_states=all_hidden_states,
        )


    def forward_core_h(self, x: torch.Tensor, video_frame=-1, return_hidden=False, return_all_hidden=False):
        '''
        新方法，将SiM的forward搬出来，每层ViT直接输入SiM，不同route都输出到最后
        '''
        # ======================================  Side 输入预处理 =============================================================
        # x_side = self.side_mamba_v.patch_embed(rearrange(x, '(b t) c h w -> b c t h w', t=self.T), channel_first=True)
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)
        x_side = self.side_mamba_v.side_pre_bn3d(self.side_mamba_v.side_conv1(x_side))
        x_side = rearrange(x_side,'b c t h w -> (b t) (h w) c').contiguous()


        BT, L, C = x_side.shape  # BT L C

        ## 为 side mamba 添加cls token， 并将vision transformer 中的cls token 调整到与 mamba 相同的位置
        if self.side_mamba_v.if_cls_token:
            cls_token = self.side_mamba_v.cls_token.expand(BT, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            self.side_mamba_v.token_position = 0
            x_side = torch.cat((cls_token, x_side), dim=1)
            L = x_side.shape[1]

        ## pos embedding
        if self.side_mamba_v.if_abs_pos_embed:
            x_side = self.side_mamba_v.PE.add_pos_embed(x_side)
        # ======================================  Backbone 输入预处理 =============================================================
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # ======================================  分层计算  =============================================================
        x = x.permute(1, 0, 2).contiguous()  # BT N D -> N BT D

        hierarchical_output = []
        for i in range(self.layers):
            x = self.transformer.resblocks[i](x)
            if i in self.side_layer_index:
                j = self.side_layer_index.index(i)

                # 将 x_side 和 x_trans 进行拼接
                xs2xt = x.permute(1, 0, 2).contiguous()
                xs2xt = self.side_mamba_v.side_linears[j](self.side_mamba_v.side_lns[j](xs2xt))

                # Original fixed Side/CLIP fusion:
                x_side = x_side * 0.5 + xs2xt * 0.5

                x_cls = x_side[..., 0, :].unsqueeze(1)  # BT 1 C
                x_hidden = x_side[..., 1:, :]

                layer = self.side_mamba_v.blocks[j]

                if self.side_mamba_v.cls_interaction == 'v1':
                    ### v1  cls不参与block的计算
                    x_hidden = layer(x_hidden, flattened=True)

                elif self.side_mamba_v.cls_interaction == 'v2':
                    ### v2  将CLS Token广播到每个空间位置，并与图像特征相加，注入全局信息，再通过全局池化提取CLS信息
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    x_cls_n = torch.mean(x_hidden, dim=1, keepdim=True)
                    x_cls = (x_cls + x_cls_n) / 2.0


                elif self.side_mamba_v.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
                    ### v3  将CLS Token广播到每个空间位置，再通过空间注意力聚合获得CLS
                    ### v4  将CLS Token广播到每个空间位置，再通过多尺度卷积融合获得CLS
                    ### 使用不同膨胀率（dilation）的卷积核，捕获不同尺度的上下文信息。
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    x_cls_n = self.side_mamba_v.cls_agg[j](x_hidden, flattened=True)
                    x_cls = (x_cls + x_cls_n) / 2.0
                x_side = torch.cat([x_cls, x_hidden], dim=-2)

                if i in self.side_layer_route_index:
                    hierarchical_output.append(x_side)


        # # v1，每层输出先平均得到 x_side ，然后再与vit相加
        # x_side = torch.stack(hierarchical_output).mean(dim=0, keepdim=False)

        # v2，每层输出有一个参数控制，加权得到 x_side ，然后再与vit相加
        x_side = torch.stack(hierarchical_output)
        norm_weights = F.softmax(self.side_mamba_v.hierarchical_weight, dim=-1)
        view_shape = (-1,) + (1,) * (x_side.ndim - 1)
        norm_weights_reshaped = norm_weights.view(view_shape)

        x_side = (x_side * norm_weights_reshaped).sum(dim=0)

        # x = x.permute(1, 0, 2).contiguous()  # N BT D -> BT N D
        x = rearrange(x, 'l (b t) d -> b t l d', t=self.T) # N BT D -> B T N D
        x_side = rearrange(x_side, '(b t) l d -> b t l d', t=self.T)

        x_side = self.side_mamba_v.side_post_ln(x_side)

        ## 已将 cls 放到 x_side 的第 0 位
        side_output = x_side @ self.side_proj
        x = x + side_output

        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x) @ self.proj
        x_cls = x[..., 0, :]
        return EncoderOutput(
            pooler_output=x_cls,
            last_hidden_state=x,
        )

    def forward(self, x: torch.Tensor, video_frame=-1, return_hidden=False, return_all_hidden=False):
        if self.side_mamba_v is not None:
            if self.side_mamba_v.hierarchical:
                forward_core = self.forward_core_h
            else:
                forward_core = self.forward_core_n
        else:
            forward_core = self.forward_core_f
        return forward_core(x, video_frame, return_hidden, return_all_hidden)

class VisualTransformer_Recongition(nn.Module):
    def __init__(self, img_size=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512, linear_patch='2d', T=12, side_network: Union[SideVMamba]=None):
        super().__init__()
        self.img_size = img_size
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.T = T
        self.layers = layers
        self.transformer = Transformer(width, layers, heads)

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)
        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


        self.side_mamba_v = side_network
        if side_network is not None:
            self.side_layer_index, self.side_layer_route_index = self.side_mamba_v.set_side_layer_index(self.layers)
            side_scale = self.side_mamba_v.side_dim ** -0.5 # (output_dim+self.side_dim) ** -0.5
            self.side_proj = nn.Parameter(side_scale * torch.zeros(self.side_mamba_v.side_dim, width))


    def forward_v1(self, x: torch.Tensor):
        '''
        去掉最后的proj，输出 width 维度的向量
        '''
        # ======================================  Side 输入预处理 =============================================================
        # x_side = self.side_mamba_v.patch_embed(rearrange(x, '(b t) c h w -> b c t h w', t=self.T), channel_first=True)
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)
        x_side = self.side_mamba_v.side_pre_bn3d(self.side_mamba_v.side_conv1(x_side))
        x_side = rearrange(x_side,'b c t h w -> (b t) (h w) c').contiguous()


        BT, L, C = x_side.shape  # BT L C

        ## 为 side mamba 添加cls token， 并将vision transformer 中的cls token 调整到与 mamba 相同的位置
        if self.side_mamba_v.if_cls_token:
            cls_token = self.side_mamba_v.cls_token.expand(BT, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            self.side_mamba_v.token_position = 0
            x_side = torch.cat((cls_token, x_side), dim=1)
            L = x_side.shape[1]

        ## pos embedding
        if self.side_mamba_v.if_abs_pos_embed:
            x_side = self.side_mamba_v.PE.add_pos_embed(x_side)

            # x_side = rearrange(x_side, '(b t) l c -> b t l c', t=self.T).contiguous()
            # x_side = x_side + self.PE.pos_embed

            # x_side = rearrange(x_side, 'b t l c -> (b t) l c').contiguous()
            # x_side = self.pos_drop(x_side)

        # ======================================  Backbone 输入预处理 =============================================================
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # ======================================  分层计算  =============================================================

        x = x.permute(1, 0, 2).contiguous()  # BT N D -> N BT D

        for i in range(self.layers):
            x = self.transformer.resblocks[i](x)
            if i in self.side_layer_index:
                j = self.side_layer_index.index(i)

                # 将 x_side 和 x_trans 进行拼接
                xs2xt = x.permute(1, 0, 2).contiguous()
                xs2xt = self.side_mamba_v.side_linears[j](self.side_mamba_v.side_lns[j](xs2xt))

                x_side = x_side * 0.5 + xs2xt * 0.5

                x_cls = x_side[..., 0, :].unsqueeze(1)  # BT 1 C
                x_hidden = x_side[..., 1:, :]

                layer = self.side_mamba_v.blocks[j]

                if self.side_mamba_v.cls_interaction == 'v1':
                    ### v1  cls不参与block的计算
                    x_hidden = layer(x_hidden, flattened=True)

                elif self.side_mamba_v.cls_interaction == 'v2':
                    ### v2  将CLS Token广播到每个空间位置，并与图像特征相加，注入全局信息，再通过全局池化提取CLS信息
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    x_cls_n = torch.mean(x_hidden, dim=1, keepdim=True)
                    x_cls = (x_cls + x_cls_n) / 2.0


                elif self.side_mamba_v.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
                    ### v3  将CLS Token广播到每个空间位置，再通过空间注意力聚合获得CLS

                    ### v4  将CLS Token广播到每个空间位置，再通过多尺度卷积融合获得CLS
                    ### 使用不同膨胀率（dilation）的卷积核，捕获不同尺度的上下文信息。
                    x_hidden = x_hidden + 0.5 * x_cls    # BT L C
                    x_hidden = layer(x_hidden, flattened=True)
                    # x_cls_n = self.cls_agg(x, flattened=True)
                    x_cls_n = self.side_mamba_v.cls_agg[j](x_hidden, flattened=True)

                    # cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
                    #
                    # cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
                    #
                    # x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)

                    x_cls = (x_cls + x_cls_n) / 2.0

                x_side = torch.cat([x_cls, x_hidden], dim=-2)

        # x = x.permute(1, 0, 2).contiguous()  # N BT D -> BT N D
        x = rearrange(x, 'l (b t) d -> b t l d', t=self.T) # N BT D -> B T N D
        x_side = rearrange(x_side, '(b t) l d -> b t l d', t=self.T)

        x_side = self.side_mamba_v.side_post_ln(x_side)

        ## 已将 cls 放到 x_side 的第 0 位
        side_output = x_side @ self.side_proj
        x = x + side_output

        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x)
        x_cls = x[..., 0, :]
        return EncoderOutput(
            pooler_output=x_cls,
            last_hidden_state=x,
        )

    def forward(self, x):
        return self.forward_v1(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 # T: int = 12,
                 # side_dim: int = 320,
                 side_network_v=None,
                 side_network_t=None,

                 ):
        super().__init__()
        self.T = side_network_v.T

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                img_size=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                T = self.T,
                # side_dim=side_dim,
                side_network=side_network_v
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )


        self.side_mamba_t = side_network_t
        if self.side_mamba_t is not None:
            side_scale_t = self.side_mamba_t.side_dim ** -0.5  # (output_dim+self.side_dim) ** -0.5
            self.side_proj_t = nn.Parameter(side_scale_t * torch.zeros(self.side_mamba_t.side_dim, transformer_width))


        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        # if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
        #     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])
        model_path = '/home/tzj/pretrained_models/clip'
        model_path = os.path.join(model_path, _PT_NAME[pretrained_clip_name])

        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"] and os.path.exists(model_path):
        # if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name], Path(model_path).parent)
                # model_path = download_model(pretrained_clip_name)   # through this
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        # return_hidden : ['all', 'last', False]
        # pooler_output: (B T) D     last_hidden_state: (B T) L D
        visual_output = self.visual(image.type(self.dtype), video_frame=video_frame, return_hidden=return_hidden)

        if return_hidden:
            return EncoderOutput(
                pooler_output=visual_output.pooler_output,
                last_hidden_state=visual_output.last_hidden_state,
                hidden_states=visual_output.hidden_states
            )
        else:
            return EncoderOutput(
                pooler_output=visual_output.pooler_output,
                # last_hidden_state=visual_output.last_hidden_state,
            )

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        # x = x.permute(1, 0, 2).contiguous()  # NLD -> LND
        encoded = self.transformer(x.permute(1, 0, 2).contiguous())

        last_hidden = encoded.last_hidden_state
        last_hidden = last_hidden.permute(1, 0, 2).contiguous()  # LND -> NLD
        all_hidden = encoded.hidden_states
        # 检查所有张量形状是否相同
        shapes = [t.shape for t in all_hidden]
        assert all(s == shapes[0] for s in shapes), "张量形状不一致！"
        all_hidden = torch.stack(all_hidden).permute(0, 2, 1, 3).contiguous()

        if self.side_mamba_t is not None:
            side_output = self.side_mamba_t(x, all_hidden)
            x_side_hidden = side_output.last_hidden_state  # B T N D
            x_side_hidden = x_side_hidden @ self.side_proj_t

            last_hidden = last_hidden + x_side_hidden


        x_hidden = self.ln_final(last_hidden).type(self.dtype) @ self.text_projection   # [batch_size, max word, d_model]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_cls = x_hidden[torch.arange(x_hidden.shape[0]), text.argmax(dim=-1)]  # [batch_size, d_model]
        # all_hidden = torch.cat((all_hidden[:-1], hidden.unsqueeze(0)), dim=0)
        if return_hidden:
            return EncoderOutput(
                pooler_output=x_cls,
                last_hidden_state=x_hidden,
                hidden_states=all_hidden,
            )
        else:
            return EncoderOutput(
                pooler_output=x_cls,
                last_hidden_state=x_hidden,
            )


    @staticmethod
    def load_clip_state_dict(clip_name, clip_path):
        if not Path(clip_path).exists():
            if clip_name in _MODELS:
                clip_path = _download(_MODELS[clip_name], Path(clip_path).parent)
            elif os.path.isfile(clip_name):
                clip_path = clip_name
            else:
                raise RuntimeError(f"Model {clip_name} not found; available models = {list(_MODELS.keys())}")

        try:
            # loading JIT archive
            model = torch.jit.load(clip_path, map_location="cpu").eval()
            clip_state_dict = model.state_dict()
        except RuntimeError:
            clip_state_dict = torch.load(clip_path, map_location="cpu")

        return clip_state_dict

    @staticmethod
    def get_clip_config(state_dict, original_config):
        vit = "visual.proj" in state_dict
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        original_config['embed_dim'] = embed_dim
        original_config['image_resolution'] = image_resolution
        original_config['vision_layers'] = vision_layers
        original_config['vision_width'] = vision_width
        original_config['vision_patch_size'] = vision_patch_size
        original_config['context_length'] = context_length
        original_config['vocab_size'] = vocab_size
        original_config['transformer_width'] = transformer_width
        original_config['transformer_heads'] = transformer_heads
        original_config['transformer_layers'] = transformer_layers

        return original_config


class CLIP_Recognition(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 # T: int = 12,
                 # side_dim: int = 320,
                 side_network_v=None,
                 side_network_t=None,

                 ):
        super().__init__()

        self.T = side_network_v.T

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer_Recongition(
                img_size=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                T = self.T,
                # side_dim=side_dim,
                side_network=side_network_v
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        # pooler_output: (B T) D     last_hidden_state: (B T) L D
        visual_output = self.visual(image.type(self.dtype), video_frame=video_frame)
        if return_hidden:
            return EncoderOutput(
                pooler_output=visual_output.pooler_output,
                last_hidden_state=visual_output.last_hidden_state,
                hidden_states=visual_output.hidden_states
            )
        else:
            return EncoderOutput(
                pooler_output=visual_output.pooler_output,
                last_hidden_state=visual_output.last_hidden_state,
            )


    @staticmethod
    def load_clip_state_dict(clip_name, clip_path):
        if not Path(clip_path).exists():
            if clip_name in _MODELS:
                clip_path = _download(_MODELS[clip_name], Path(clip_path).parent)
            elif os.path.isfile(clip_name):
                clip_path = clip_name
            else:
                raise RuntimeError(f"Model {clip_name} not found; available models = {list(_MODELS.keys())}")

        try:
            # loading JIT archive
            model = torch.jit.load(clip_path, map_location="cpu").eval()
            clip_state_dict = model.state_dict()
        except RuntimeError:
            clip_state_dict = torch.load(clip_path, map_location="cpu")

        # state_dict = {}
        # for key, val in clip_state_dict.items():
        #     new_key = "clip." + key
        #     if new_key not in state_dict:
        #         state_dict[new_key] = val.clone()

        return clip_state_dict

    @staticmethod
    def get_clip_config(state_dict, original_config):
        vit = "visual.proj" in state_dict
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        original_config['embed_dim'] = embed_dim
        original_config['image_resolution'] = image_resolution
        original_config['vision_layers'] = vision_layers
        original_config['vision_width'] = vision_width
        original_config['vision_patch_size'] = vision_patch_size
        original_config['context_length'] = context_length
        original_config['vocab_size'] = vocab_size
        original_config['transformer_width'] = transformer_width
        original_config['transformer_heads'] = transformer_heads
        original_config['transformer_layers'] = transformer_layers

        return original_config


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

