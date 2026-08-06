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
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from einops import rearrange
from random import sample
import torch.utils.checkpoint as checkpoint
import numpy as np
from dataclasses import dataclass

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
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


def _download(url: str, root: Union[str, Path] = os.path.expanduser("~/.cache/clip")):
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


# =============================
@dataclass
class EncoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        global_feature (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
        The overall representation of each sequence.

        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

    """
    global_feature: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'padding={}'.format(self.drop_prob)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

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
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
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


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


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
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = nn.MultiheadAttention(dim, dim // 64, dropout=0.)
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
                nn.init.normal_(p, std=side_attn_std)
            elif 'attn.out_proj.weight' in name:
                nn.init.normal_(p, std=side_proj_std)

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
        self.attn_mask = None  # self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def shift_token(self, x_token):  # [1, bt, dilation]
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
        xt = xt.permute(1, 0, 2)
        xt = xt + side_position_embeddings.to(x.dtype)
        xt = xt.permute(1, 0, 2)
        xt = self.attention(self.ln_1(xt))
        x = x + xt[1:, :, :]

        x_ = x
        x = rearrange(x, '(h w) (b t) dilation -> b dilation t h w', h=h, t=self.T)
        x = self.norm2(x)
        x = rearrange(x, 'b dilation t h w -> (h w) (b t) dilation', h=h, t=self.T)
        x = x_ + self.drop_path(self.mlp(x))
        return x


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0.0):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
#         self.ln_1 = LayerNorm(d_model)
#         self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#
#     def attention(self, x: torch.Tensor, x_mask:torch.Tensor):
#         if x_mask is not None:
#             x_mask = x_mask.to(dtype=torch.bool, device=x.device)
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]
#
#     def forward(self, x: torch.Tensor, x_mask:torch.Tensor=None):
#         x = x + self.attention(self.ln_1(x), x_mask)
#         x = x + self.mlp(self.ln_2(x))
#         return x

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

    def attention(self, x: torch.Tensor, x_mask:torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    # def attention(self, x: torch.Tensor, x_mask:torch.Tensor):
    #     if x_mask is not None:
    #         x_mask = x_mask.to(dtype=torch.bool, device=x.device)
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def forward(self, x: torch.Tensor, x_mask:torch.Tensor=None, use_checkpoint=False):
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1.float()(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1.float()(x), x_mask))

        # FFN
        if use_checkpoint:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2.float()(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2.float()(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None, dropout=0.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, output_hidden_states: bool = True):
        all_hidden_states = () if output_hidden_states else None
        for block in self.resblocks:
            x = block(x, x_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        return EncoderOutput(
            last_hidden_state = x,
            hidden_states = all_hidden_states)


class SideTransformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 output_dim,
                 dropout=None,
                 side_dim=384,
                 max_frames=8,
                 input_resolution=224,
                 patch_size=32,
                 linear_patch: str = '3d', side_layers_mode: str = 'all'):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)]
        print('dropout used:{}'.format(dropout))
        patch_num = (input_resolution // patch_size) ** 2
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.ln_post = LayerNorm(width)
        self.width = width
        self.layers = layers
        self.max_frames = max_frames

        self.side_transformer = []
        self.side_linears = []
        self.side_lns = []

        self.side_dim = side_dim

        self.side_layers_mode = side_layers_mode  # all, interval, top
        if self.side_layers_mode == 'all':
            self.side_layers = [i for i in range(layers)]
        elif self.side_layers_mode == 'interval':
            self.side_layers = [i + 1 for i in range(0, layers, 2)]
        elif self.side_layers_mode == 'top':
            self.side_layers = [i for i in range(layers // 2, layers)]
        else:
            raise NotImplementedError

        for i in range(layers):
            if i in self.side_layers:
                self.side_transformer.append(AttnCBlock(self.side_dim, self.side_dim, kernel_size=1, T=self.max_frames))
                self.side_linears.append(nn.Linear(width, self.side_dim))
                self.side_lns.append(LayerNorm(width))
        self.side_lns = nn.ModuleList(self.side_lns)

        self.side_linears = nn.ModuleList(self.side_linears)
        self.side_transformer = nn.ModuleList(self.side_transformer)
        side_scale = self.side_dim ** -0.5
        self.side_spatial_position_embeddings = nn.Parameter(side_scale * torch.randn((patch_num + 1, self.side_dim)))
        nn.init.normal_(self.side_spatial_position_embeddings, std=0.01)


        side_scale = self.side_dim ** -0.5  # (output_dim+self.side_dim) ** -0.5
        self.side_proj = nn.Parameter(side_scale * torch.zeros(self.side_dim, width))
        self.side_post_bn = bn_3d(self.side_dim)
        self.side_conv1 = conv_3xnxn_std(3, self.side_dim, kernel_size=patch_size, stride=patch_size)
        self.side_pre_bn3d = nn.BatchNorm3d(self.side_dim)
        nn.init.ones_(self.side_pre_bn3d.weight)
        nn.init.zeros_(self.side_pre_bn3d.bias)

    def forward(self, x: torch.Tensor, vit_outputs: EncoderOutput):
        '''
        :param x: The raw image features [bs, max_frames, 3, 224, 224]
        :param vit_outputs:  Output features from Visual Encoder,   subsequent elements represent the hidden states [bs, max_frames, patches, hidden_size] of the Visual Encoder output   at each layer.
        :param video_frame:
        :return:
        '''
        from einops import rearrange
        # x = inputs.channel_feature
        visual_hidden_states = vit_outputs.hidden_states

        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.max_frames)
        x_side = self.side_pre_bn3d(self.side_conv1(x_side))
        x_side = rearrange(x_side, 'b c t h w -> (b t) (h w) c')
        x_side = x_side.permute(1, 0, 2)

        k = 0
        for i in range(len(visual_hidden_states)):
            # x = self.resblocks[i](x)
            if i not in self.side_layers:
                continue
            xs2xt = self.side_linears[k](self.side_lns[k](visual_hidden_states[i]))
            x_token = xs2xt[:1, :, :]
            xs2xt = xs2xt[1:, :, :]
            x_side = 0.5 * x_side + 0.5 * xs2xt
            x_side = self.side_transformer[k](x_side, x_token, self.side_spatial_position_embeddings, i)
            k += 1

        x_side = x_side.permute(1, 0, 2)  # LND -> NLD

        h = int(x_side.shape[1] ** 0.5)
        x_side = rearrange(x_side, '(b t) (h w) dilation -> b dilation t h w', t=self.max_frames, h=h)
        x_side = self.side_post_bn(x_side)

        x_side = x_side.permute(0, 2, 1, 3, 4)

        x_side = x_side.flatten(3).mean(-1)
        x_side = x_side @ self.side_proj.to(x_side.dtype)

        x = vit_outputs.last_hidden_state

        x = rearrange(x[:, 0, :], '(b t) dilation -> b t dilation', t=self.max_frames)
        x = x + x_side
        # Move the three lines below to `encode_image` for entire hidden sequence
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)


    def forward(self, x: torch.Tensor, x_mask, video_frame=-1, output_hidden_states=True):
        all_hidden_states = () if output_hidden_states else None

        # recode the raw image feature x: [bs, max_frames, 3, 224, 224]
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (x,)

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)  # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)  # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2],
                             x_3d.shape[-1]).contiguous()  # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t=self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x, x_mask)
        encoded = self.transformer(x, x_mask, output_hidden_states=output_hidden_states)

        x = encoded.last_hidden_state.permute(1, 0, 2)
        x = self.ln_post(x)

        if output_hidden_states:
            all_hidden_states = encoded.hidden_states
            # all_hidden_states = all_hidden_states.permute(1, 0, 2)
            # all_hidden_states = self.ln_post(all_hidden_states)

        # x, all_hidden_states = self.transformer(x, x_mask).last_hidden_state, self.transformer(x, x_mask).hidden_states
        # x, all_hidden_states = x.permute(1, 0, 2), all_hidden_states.permute(1, 0, 2)  # LND -> NLD


        # if self.proj is not None:
        #     x = x @ self.proj

        return EncoderOutput(
            last_hidden_state = x,
            hidden_states = all_hidden_states)

class CLIPEncoder(nn.Module):
    '''
    预训练的多模态编码器，同时加载文本和图像编码器
    '''
    def __init__(self, embed_dim: int,
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
                 T: int = 12,
                 side_dim: int = 320,
                 **kwargs):
        super(CLIPEncoder, self).__init__()
        self.context_length = context_length


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
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                # max_frames = max_frames,
                # side_dim=side_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )



        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        # self.initialize_parameters()

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

    def encode_image(self, image, mask, video_frame=-1):
        x = self.visual(image.type(self.dtype), mask,  video_frame=video_frame)
        return x
        # hidden = self.visual.ln_post(hidden) @ self.visual.proj

        # x = hidden[:, 0, :]

        # if return_hidden:
        #     return x, hidden

        # return EncoderOutput(
        #     channel_feature=x.channel_feature,
        #     last_hidden_state=x.last_hidden_state,
        #     hidden_states=x.hidden_states,
        # )

    def encode_text(self, text, output_hidden_states=True):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        encoded = self.transformer(x, output_hidden_states=output_hidden_states)
        x = encoded.last_hidden_state.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection  # [batch_size, max word, d_model]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]  # [batch_size, d_model]


        return EncoderOutput(
            global_feature=x,
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
        )

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


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


# def build_model(state_dict: dict):
#     vit = "visual.proj" in state_dict
#
#     if vit:
#         vision_width = state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len(
#             [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
#                         [1, 2, 3, 4]]
#         vision_layers = tuple(counts)
#         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         vision_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32
#
#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
#
#     model = CLIP(
#         embed_dim,
#         image_resolution, vision_layers, vision_width, vision_patch_size,
#         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
#     )
#
#     for key in ["input_resolution", "context_length", "vocab_size"]:
#         if key in state_dict:
#             del state_dict[key]
#
#     convert_weights(model)
#     model.load_state_dict(state_dict)
#     return model.eval()
