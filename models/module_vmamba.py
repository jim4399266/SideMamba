import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, to_2tuple
from einops import rearrange

from src.models.module_utils import EncoderOutput
from src.models.module_pos import PositionEmbedding

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
from src.models.csm import create_route_selector
from src.models.csm.csm_triton_o import cross_scan_fn, cross_merge_fn
# from src.models.csm.sequential_scan import SequentialScanFn
from src.models.csm.csms6s import selective_scan_fn
from src.models.aggregator import ClsAggregator
from src.models.module_fft import VideoFFParser, Spectral_Layer


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# =====================================================

# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()  # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

# ====================================================
class SS3Dv0:
    def __initv0_3d__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v0",
            channel_first=False,
            select_type="s",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            merge_mean=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.k_group = 12
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv0_3d

        # tags for forward_type ==============================
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        self.oact = False
        self.disable_z = if_noz
        self.select_type = select_type
        self.route_type = route_type
        self.step_size = step_size

        self.if_bidirectional = if_bidirectional
        self.merge_mean = merge_mean

        # k_group = 4 if route_type not in ['sy', 'synthetic'] else 12
        # self.k_group = k_group if if_bidirectional else k_group / 2

        self.forward_core = partial(self.forward_corev0_3d, force_fp32=False, no_einsum=True)
        # self.RS = RouteSelector(select_type=select_type, route_type=route_type, step_size=self.step_size, bidirectional=if_bidirectional, merge_mean=merge_mean)
        self.RS = create_route_selector(select_type=select_type, route_type=route_type, step_size=self.step_size, bidirectional=if_bidirectional, merge_mean=merge_mean)
        self.k_group = self.RS.K
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev0_3d(
            self,
            x: torch.Tensor = None,  # B D T H W
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)

        delta_softplus = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        xs = self.RS.scan(x)  # [b c t h w] -> [b k c t*h*w]

        K, D, R, N = self.k_group, self.d_inner, self.dt_rank, self.d_state
        [B, K, C, L] = xs.shape

        x_proj_bias = getattr(self, "x_proj_bias", None)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            # x_dbl = torch.einsum("b k dilation l, k c dilation -> b k c l", xs, self.x_proj_weight)
            # if x_proj_bias is not None:
            #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            # dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # if hasattr(self, "dt_projs_weight"):
            #     dts = torch.einsum("b k r l, k dilation r -> b k dilation l", dts, self.dt_projs_weight)
            raise NotImplementedError

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        y = self.RS.merge(ys)

        return y.to(x.dtype)

    def forwardv0_3d(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, t, h, w, dilation)
            # if not self.disable_z_act:
            #     z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.with_dconv:
            x = self.conv3d(x)  # (b, dilation, t, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        if not self.channel_first:
            y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = self.out_norm(y)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

class SS3Dv3:
    def __initv0_3d__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v3",
            channel_first=False,
            select_type="s",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            merge_mean=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.k_group = 12
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv3_3d

        # tags for forward_type ==============================
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        self.oact = False
        self.disable_z = if_noz
        self.select_type = select_type
        self.route_type = route_type
        self.step_size = step_size

        self.if_bidirectional = if_bidirectional
        self.merge_mean = merge_mean

        # k_group = 4 if route_type not in ['sy', 'synthetic'] else 12
        # self.k_group = k_group if if_bidirectional else k_group / 2

        self.forward_core = partial(self.forward_corev3_3d, force_fp32=False, no_einsum=True)
        # self.RS = RouteSelector(select_type=select_type, route_type=route_type, step_size=self.step_size, bidirectional=if_bidirectional, merge_mean=merge_mean)
        self.RS = create_route_selector(select_type=select_type, route_type=route_type, step_size=self.step_size, bidirectional=if_bidirectional, merge_mean=merge_mean)
        self.k_group = self.RS.K
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev3_3d(
            self,
            x: torch.Tensor = None,  # B D T H W
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)

        delta_softplus = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        xs = self.RS.scan(x)  # [b c t h w] -> [b k c t*h*w]

        K, D, R, N = self.k_group, self.d_inner, self.dt_rank, self.d_state
        [B, K, C, L] = xs.shape

        x_proj_bias = getattr(self, "x_proj_bias", None)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            # x_dbl = torch.einsum("b k dilation l, k c dilation -> b k c l", xs, self.x_proj_weight)
            # if x_proj_bias is not None:
            #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            # dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # if hasattr(self, "dt_projs_weight"):
            #     dts = torch.einsum("b k r l, k dilation r -> b k dilation l", dts, self.dt_projs_weight)
            raise NotImplementedError

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        # v0 的 selective_scan 似乎有问题，
        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        y = self.RS.merge(ys)

        return y.to(x.dtype)

    def forwardv3_3d(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, t, h, w, dilation)
            # if not self.disable_z_act:
            #     z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.with_dconv:
            x = self.conv3d(x)  # (b, dilation, t, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        if not self.channel_first:
            y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = self.out_norm(y)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class SS3Dv1:
    def __initv1_3d__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            grid_size=7,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v0",
            channel_first=False,
            select_type="s",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            merge_mean=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.k_group = 12
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv1_3d

        # tags for forward_type ==============================
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        self.oact = False
        self.disable_z = if_noz
        self.select_type = select_type
        self.route_type = route_type
        self.step_size = step_size

        self.if_bidirectional = if_bidirectional
        self.merge_mean = merge_mean

        # k_group = 4 if route_type not in ['sy', 'synthetic'] else 12
        # self.k_group = k_group if if_bidirectional else k_group / 2
        self.RS = create_route_selector(select_type=select_type, route_type=route_type, bidirectional=if_bidirectional,
                                merge_mean=merge_mean)
        self.k_group = self.RS.K
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        self.fft = VideoFFParser(d_proj, 12, grid_size, grid_size)
        self.forward_core = partial(self.forward_corev1_3d, force_fp32=False, no_einsum=True)

        # conv =======================================
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev1_3d(
            self,
            x: torch.Tensor = None,  # B D T H W
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)

        delta_softplus = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        xs = self.RS.scan(x)  # [b c t h w] -> [b k c t*h*w]

        K, D, R, N = self.k_group, self.d_inner, self.dt_rank, self.d_state
        [B, K, C, L] = xs.shape

        x_proj_bias = getattr(self, "x_proj_bias", None)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            # x_dbl = torch.einsum("b k dilation l, k c dilation -> b k c l", xs, self.x_proj_weight)
            # if x_proj_bias is not None:
            #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            # dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # if hasattr(self, "dt_projs_weight"):
            #     dts = torch.einsum("b k r l, k dilation r -> b k dilation l", dts, self.dt_projs_weight)
            raise NotImplementedError

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        y = self.RS.merge(ys)

        return y.to(x.dtype)

    def forwardv1_3d(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, t, h, w, dilation)
            # if not self.disable_z_act:
            #     z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.with_dconv:
            x = self.conv3d(x)  # (b, dilation, t, h, w)
        x = self.act(x)
        y1 = self.forward_core(x)
        y2 = self.fft(x)
        if not self.channel_first:
            y1 = y1.permute(0, 2, 3, 4, 1).contiguous()
            y2 = y2.permute(0, 2, 3, 4, 1).contiguous()

        y = self.out_norm(y1+y2)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class SS3Dv2:
    def __initv2_3d__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            grid_size=7,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v0",
            channel_first=False,
            select_type="s",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            merge_mean=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.k_group = 12
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2_3d

        # tags for forward_type ==============================
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        self.oact = False
        self.disable_z = if_noz
        self.select_type = select_type
        self.route_type = route_type
        self.step_size = step_size

        self.if_bidirectional = if_bidirectional
        self.merge_mean = merge_mean

        # k_group = 4 if route_type not in ['sy', 'synthetic'] else 12
        # self.k_group = k_group if if_bidirectional else k_group / 2
        self.RS = create_route_selector(select_type=select_type, route_type=route_type, bidirectional=if_bidirectional,
                                merge_mean=merge_mean)
        self.k_group = self.RS.K
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        self.fft = VideoFFParser(d_proj, 12, grid_size, grid_size)
        self.forward_core = partial(self.forward_corev2_3d, force_fp32=False, no_einsum=True)

        # conv =======================================
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2_3d(
            self,
            x: torch.Tensor = None,  # B D T H W
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)

        delta_softplus = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        xs = self.RS.scan(x)  # [b c t h w] -> [b k c t*h*w]

        K, D, R, N = self.k_group, self.d_inner, self.dt_rank, self.d_state
        [B, K, C, L] = xs.shape

        x_proj_bias = getattr(self, "x_proj_bias", None)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            # x_dbl = torch.einsum("b k dilation l, k c dilation -> b k c l", xs, self.x_proj_weight)
            # if x_proj_bias is not None:
            #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            # dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # if hasattr(self, "dt_projs_weight"):
            #     dts = torch.einsum("b k r l, k dilation r -> b k dilation l", dts, self.dt_projs_weight)
            raise NotImplementedError

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        y = self.RS.merge(ys)

        return y.to(x.dtype)

    def forwardv2_3d(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, t, h, w, dilation)
            # if not self.disable_z_act:
            #     z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.with_dconv:
            x = self.conv3d(x)  # (b, dilation, t, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.fft(y)
        if not self.channel_first:
            y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = self.out_norm(y)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
# ========= for test =======
class SS3Dt1:
    def __initt1_3d__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="t0",
            channel_first=False,
            select_type="e",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            merge_mean=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardt1_3d

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.merge_mean, forward_type = checkpostfix("_mm", forward_type)

        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        self.oact = False
        self.disable_z = if_noz
        self.select_type = select_type
        self.route_type = route_type
        self.step_size = step_size

        self.if_bidirectional = if_bidirectional
        self.merge_mean = merge_mean

        k_group = 4 if route_type not in ['sy', 'synthetic'] else 12
        self.k_group = k_group if if_bidirectional else k_group / 2
        # forward_type debug =======================================

        self.forward_core = partial(self.forward_coret1_3d, force_fp32=False, no_einsum=True)

        self.RS = create_route_selector(select_type=select_type, route_type=route_type, bidirectional=if_bidirectional, merge_mean=merge_mean)
        # self.RS = build_ssm_route(select_type=select_type, route_type=route_type, step_size=self.step_size, bidirectional=if_bidirectional, merge_mean=merge_mean)

        # SELECTIVE_SCAN_TYPES = dict(
        #     t1_3d=['v1', 12],  # full bi-directional scan
        #     t12_3d=['v2', 6],  # forward uni-directional scan
        #     t13_3d=['v3', 8],  # forward uni-directional scan
        # )

        # self.scan_fn, self.merge_fn = SELECTIVE_SCAN_TYPES[forward_type]
        # self.scan_route_type, self.k_group = SELECTIVE_SCAN_TYPES[forward_type]

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_coret1_3d(
            self,
            x: torch.Tensor = None,  # B D T H W
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)


        xs = self.RS.scan(x)

        K, D, R, N = self.k_group, self.d_inner, self.dt_rank, self.d_state

        [B, K, C, L] = xs.shape
        # L = T * H * W

        x_proj_bias = getattr(self, "x_proj_bias", None)

        # xs = cross_3d_scan_fn(x, K, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            # x_dbl = torch.einsum("b k dilation l, k c dilation -> b k c l", xs, self.x_proj_weight)
            # if x_proj_bias is not None:
            #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            # dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            # if hasattr(self, "dt_projs_weight"):
            #     dts = torch.einsum("b k r l, k dilation r -> b k dilation l", dts, self.dt_projs_weight)
            raise NotImplementedError

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # ys: torch.Tensor = selective_scan(
        #     xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        # ).view(B, K, -1, T, H, W)

        ys = xs.view(B, K, -1, L)

        # y: torch.Tensor = cross_3d_merge_fn(ys, K, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
        #                                  force_torch=scan_force_torch, scan_route_type=self.scan_route_type)
        y = self.RS.merge(ys)

        # if self.merge_mean:
        #     y = y / float(self.k_group)


        # y = y.view(B, -1, T, H, W)

        # if not channel_first:
        #     y = y.view(B, -1, T * H * W).transpose(dim0=1, dim1=2).contiguous().view(B, T, H, W, -1)  # (B, L, C)



        # y = out_norm(y)

        return y.to(x.dtype)

    def forwardt1_3d(self, x: torch.Tensor, **kwargs):
        # x = self.in_proj(x)
        # if not self.disable_z:
        #     x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, t, h, w, dilation)
        #     if not self.disable_z_act:
        #         z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        # if self.with_dconv:
        #     x = self.conv3d(x)  # (b, dilation, t, h, w)
        # x = self.act(x)
        y = self.forward_core(x)
        if not self.channel_first:
            y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = self.out_norm(y)
        y = self.out_act(y)
        # if not self.disable_z:
        #     y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

class SSBlock(nn.Module, SS3Dv0, SS3Dv1, SS3Dv2, SS3Dt1, SS2Dv0, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            grid_size=7,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="3d_v0",
            channel_first=False,
            select_type="e",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            if_divide_out=True,
            # ======================
            **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, grid_size=grid_size, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, select_type=select_type, route_type = route_type, step_size = step_size,
                 if_noz = if_noz, if_bidirectional = if_bidirectional, if_divide_out = if_divide_out, channel_first=channel_first,
        )
        if '3d' in forward_type:
            if forward_type.startswith("m"):
                self.__initm0_3d__(**kwargs)
            elif forward_type.startswith("v0"):
                self.__initv0_3d__(**kwargs)
            elif forward_type.startswith("v1"):
                self.__initv1_3d__(**kwargs)
            elif forward_type.startswith("v2"):
                self.__initv2_3d__(**kwargs)
            elif forward_type.startswith("t0"):
                self.__initt0_3d__(**kwargs)
            elif forward_type.startswith("t1"):
                self.__initt1_3d__(**kwargs)
            else:
                raise NotImplementedError(forward_type)
        else:
            if forward_type.startswith("m"):
                self.__initm0__(**kwargs)
            elif forward_type.startswith("v"):
                self.__initv2__(**kwargs)
            elif forward_type.startswith("t"):
                self.__initt0__(**kwargs)
            else:
                raise NotImplementedError(forward_type)

class VSSBlock_o(nn.Module):
    '''
    2025-12-17:添加傅里叶变换之前的代码
    '''
    def __init__(
            self,
            hidden_dim: int = 0,
            T=12,
            grid_size=7,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v0",
            select_type="e",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            if_divide_out=True,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS: type = SSBlock,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.T = T
        self.forward_type = forward_type

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                select_type=select_type,
                route_type=route_type,
                step_size=step_size,
                if_noz=if_noz,
                if_bidirectional=if_bidirectional,
                if_divide_out=if_divide_out,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, x: torch.Tensor, flattened=False):
        if flattened:
            if '3d' in self.forward_type:
                assert self.T > 0
                # 3d卷积，输入的x包含5个维度  b t h w c
                x = rearrange(x, '(b t) (h w) c -> b t h w c', t=self.T, h=int(math.sqrt(x.shape[-2])))
            else:
                # 2d卷积，输入的x包含4个维度  b h w c
                x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2])))


        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN


        if flattened:
            if self.T > 0:
                x = rearrange(x, 'b t h w c -> (b t) (h w) c')
            else:
                x = rearrange(x, 'b h w c -> b (h w) c')
        return x

class VSSBlock(nn.Module):
    '''
    新增傅里叶变换之后的代码
    '''
    def __init__(
            self,
            hidden_dim: int = 0,
            T=12,
            grid_size=7,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v0",
            select_type="e",
            route_type="sy",
            step_size=-1,
            if_noz=True,
            if_bidirectional=True,
            if_divide_out=True,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS: type = SSBlock,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.T = T
        self.forward_type = forward_type

        # self.fft = VideoFFParser(hidden_dim, T, grid_size, grid_size)
        # self.spectral = Spectral_Layer(T, grid_size, grid_size, hidden_dim)

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                grid_size=grid_size,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                select_type=select_type,
                route_type=route_type,
                step_size=step_size,
                if_noz=if_noz,
                if_bidirectional=if_bidirectional,
                if_divide_out=if_divide_out,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, x: torch.Tensor, flattened=False):
        if flattened:
            if '3d' in self.forward_type:
                assert self.T > 0
                # 3d卷积，输入的x包含5个维度  b t h w c
                x = rearrange(x, '(b t) (h w) c -> b t h w c', t=self.T, h=int(math.sqrt(x.shape[-2])))
            else:
                # 2d卷积，输入的x包含4个维度  b h w c
                x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[-2])))


        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))

        # if self.post_norm:   # fft，并列
        #     x = x + self.drop_path(self.norm(self.op(x) + self.spectral(x, channel_first=False)))
        # else:
        #     x = x + self.drop_path(self.op(self.norm(x)) + self.spectral(self.norm(x), channel_first=False))

        # if self.post_norm:     # fft2，先mamba，后傅里叶
        #     x = x + self.drop_path(self.norm(self.spectral(self.op(x), channel_first=False)))
        #     # x = x + self.drop_path(self.norm(self.op(x) + self.spectral(x, channel_first=False)))
        # else:
        #     x = x + self.drop_path(self.spectral(self.op(self.norm(x)), channel_first=False))
        #     # x = x + self.drop_path(self.op(self.norm(x)) + self.spectral(self.norm(x), channel_first=False))
        #
        # if self.post_norm:  # fft3， 先傅里叶，再mamba
        #     x = x + self.drop_path(self.norm(self.op(self.spectral(x, channel_first=False))))
        #     # x = x + self.drop_path(self.norm(self.op(x) + self.spectral(x, channel_first=False)))
        # else:
        #     x = x + self.drop_path(self.op(self.spectral(self.norm(x), channel_first=False)))
        #     # x = x + self.drop_path(self.op(self.norm(x)) + self.spectral(self.norm(x), channel_first=False))
        #


        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN


        if flattened:
            if self.T > 0 and x.ndim == 5:
                x = rearrange(x, 'b t h w c -> (b t) (h w) c')
            else:
                x = rearrange(x, 'b h w c -> b (h w) c')
        return x

# =====================================================


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)


class ComplexVideoPatchConv3d(nn.Module):
    """Complex 3D patch embedding using appearance and temporal difference."""

    def __init__(self, inp, oup, kernel_size=3, stride=3, groups=1):
        super().__init__()
        conv_kwargs = dict(
            in_channels=inp,
            out_channels=oup,
            kernel_size=(3, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(1, 0, 0),
            groups=groups,
            bias=False,
        )
        # These two real convolutions implement the real and imaginary kernels.
        self.real_conv = nn.Conv3d(**conv_kwargs)
        self.imag_conv = nn.Conv3d(**conv_kwargs)
        # Begin close to a real-valued convolution, then learn the complex motion interaction.
        nn.init.zeros_(self.imag_conv.weight)
        self.real_bias = nn.Parameter(torch.zeros(oup))
        self.imag_bias = nn.Parameter(torch.zeros(oup))

        # Learn how much appearance/motion information to retain before real-valued BN and Mamba.
        self.complex_fusion = nn.Conv3d(oup * 2, oup, kernel_size=1, bias=True)
        self._init_fusion()

    def _init_fusion(self):
        # Start from the real response and let training progressively incorporate the imaginary response.
        nn.init.zeros_(self.complex_fusion.weight)
        nn.init.zeros_(self.complex_fusion.bias)
        with torch.no_grad():
            channel_index = torch.arange(self.real_conv.out_channels)
            self.complex_fusion.weight[channel_index, channel_index, 0, 0, 0] = 1.0

    def forward(self, x):
        # Real part describes frame appearance; imaginary part explicitly encodes inter-frame motion.
        x_imag = torch.zeros_like(x)
        x_imag[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        # (Wr + iWi)(Xr + iXi) = (WrXr - WiXi) + i(WrXi + WiXr).
        real = self.real_conv(x) - self.imag_conv(x_imag)
        imag = self.real_conv(x_imag) + self.imag_conv(x)
        real = real + self.real_bias.view(1, -1, 1, 1, 1)
        imag = imag + self.imag_bias.view(1, -1, 1, 1, 1)

        # Return real features so existing BatchNorm3d and selective scan remain unchanged.
        return self.complex_fusion(torch.cat([real, imag], dim=1))





class PatchEmbed_3D(nn.Module):
    """ 3D Images to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768,
                 flatten=True):
        super().__init__()

        self.con3d_pre = conv_3xnxn_std(in_chans, embed_dim // 2, kernel_size=patch_size, stride=stride)
        self.con3d_post = conv_3xnxn_std(embed_dim // 2, embed_dim, kernel_size=1, stride=1)
        self.norm_pre = nn.BatchNorm3d(embed_dim // 2)
        self.norm_post = nn.BatchNorm3d(embed_dim)
        self.gelu = nn.GELU()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # nn.init.ones_(self.norm_pre.weight)
        # nn.init.zeros_(self.norm_pre.bias)
        # nn.init.ones_(self.norm_post.weight)
        # nn.init.zeros_(self.norm_post.bias)

    def forward(self, x, channel_first=True):
        if not channel_first:
            x = x.permute(0, 4, 2, 3, 1) # BTHWC -> BCTHW
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.norm_pre(self.con3d_pre(x))
        x = self.gelu(x)
        x = self.norm_post(self.con3d_post(x))
        if self.flatten:
            x = x.flatten(start_dim=-2, end_dim=-1)  #   BCTHW -> BCTN

        if not channel_first:
            x = x.permute(0, 2, 3, 1)  #  BCTN ->BTNC
        return x


#
# class SideVMamba_n(nn.Module):
#     def __init__(self,
#                  network='vmamba',
#                  img_size=224,
#                  patch_size=16,
#                  channels=3,
#                  T=12,
#                  stride=16,
#                  channel_first=False,
#                  # =========================
#                  side_layers_mode='all', # all, top, interval
#                  mamba_depth=12,
#                  layer_depth=2,
#                  side_dim=192,
#                  trans_dim=768,
#                  # =========================
#                  ssm_d_state=16,
#                  ssm_ratio=2.0,
#                  ssm_dt_rank="auto",
#                  ssm_drop_rate=0.0,
#                  ssm_act_layer="silu",
#                  ssm_conv=3,
#                  ssm_conv_bias=True,
#                  ssm_init="v0",
#                  forward_type="v0",
#                  select_type="e",
#                  route_type = "sy",
#                  step_size = -1,
#                  if_noz = True,
#                  if_bidirectional = True,
#                  if_divide_out = True,
#                  # =========================
#                  norm_epsilon=1e-5,
#                  rms_norm=True,
#                  mlp_ratio=4.0,
#                  mlp_act_layer="gelu",
#                  mlp_drop_rate=0.0,
#                  gmlp=False,
#                  # =========================
#                  drop_path_rate=0.1,
#                  norm_layer="LN",  # "BN", "LN2D"
#                  fused_add_norm=True,
#                  residual_in_fp32=True,
#                  # device=None,
#                  # dtype=None,
#                  # if_bidirectional=False,
#                  if_abs_pos_embed=True,
#                  bimamba_type="v2",
#                  if_cls_token=True,
#                  use_checkpoint=False,
#                  cls_interaction='v1',
#                  **kwargs):
#         assert network in ['vmamba', 'vmamba2']
#         # factory_kwargs = {"device": device, "dtype": dtype}
#         # add factory_kwargs into kwargs
#         # kwargs.update({
#         #      'T': T, 'stride': stride, 'channel_first': channel_first,
#         #     # =========================
#         # 'side_layers_mode': side_layers_mode, 'mamba_depth': mamba_depth, 'layer_depth': layer_depth, 'side_dim': side_dim, 'trans_dim': trans_dim,
#         #     # =========================
#         # 'ssm_d_state': ssm_d_state, 'ssm_ratio': ssm_ratio, 'ssm_dt_rank': ssm_dt_rank, 'ssm_drop_rate': ssm_drop_rate, 'ssm_act_layer': ssm_act_layer, 'ssm_conv': ssm_conv, 'ssm_conv_bias': ssm_conv_bias, 'ssm_init': ssm_init, 'forward_type': forward_type, 'select_type': select_type, 'route_type': route_type, 'step_size': step_size, 'if_noz': if_noz, 'if_bidirectional': if_bidirectional, 'if_divide_out': if_divide_out,
#         #     # =========================
#         # 'norm_epsilon': norm_epsilon,
#         # 'rms_norm': rms_norm,
#         # 'mlp_ratio': mlp_ratio,
#         # 'mlp_act_layer': mlp_act_layer,
#         # 'mlp_drop_rate': mlp_drop_rate,
#         # 'gmlp': gmlp,
#         #     # =========================
#         # 'drop_path_rate': drop_path_rate,
#         # 'norm_layer': norm_layer,  # "BN", "LN2D"
#         # 'fused_add_norm': fused_add_norm,
#         # 'residual_in_fp32': residual_in_fp32,
#         # 'if_abs_pos_embed': if_abs_pos_embed,
#         # 'bimamba_type': bimamba_type,
#         # 'if_cls_token': if_cls_token,
#         # 'use_checkpoint': use_checkpoint,
#         # 'cls_interaction': cls_interaction,
#         # })
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         # self.if_bidirectional = if_bidirectional
#         self.side_layers_mode = side_layers_mode
#
#         self.if_abs_pos_embed = if_abs_pos_embed
#
#         self.if_cls_token = if_cls_token
#
#         self.num_tokens = 1 if if_cls_token else 0
#         self.bimamba_type = bimamba_type
#         self.cls_interaction = cls_interaction
#
#         _NORMLAYERS = dict(
#             ln=nn.LayerNorm,
#             ln2d=LayerNorm2d,
#             bn=nn.BatchNorm2d,
#             bn3d=nn.BatchNorm3d
#         )
#
#         _ACTLAYERS = dict(
#             silu=nn.SiLU,
#             gelu=nn.GELU,
#             relu=nn.ReLU,
#             sigmoid=nn.Sigmoid,
#         )
#         # pretrain parameters
#         self.T = T
#         self.side_dim = side_dim  # num_features for consistency with other models
#
#         norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
#         ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
#         mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
#
#
#         self.patch_embed = PatchEmbed_3D(
#                 img_size=img_size, patch_size=patch_size, stride=patch_size, in_chans=channels, embed_dim=side_dim)
#
#         num_patches = self.patch_embed.num_patches
#
#         self.token_position = -1
#         if if_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, self.side_dim))
#
#
#         if if_abs_pos_embed:
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.side_dim))
#
#         inter_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, mamba_depth * layer_depth)]  # stochastic depth decay rule
#
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#
#         # 创建mamba blocks
#
#
#         # mamba layers
#         # self.side_layers_mode = 'all'  # all, interval, top
#         # if self.side_layers_mode == 'all':
#         #     self.side_layers = [i for i in range(mamba_depth)]
#         # elif self.side_layers_mode == 'interval':
#         #     self.side_layers = [i + 1 for i in range(0, mamba_depth, 2)]
#         # elif self.side_layers_mode == 'top':
#         #     self.side_layers = [i for i in range(mamba_depth // 2, mamba_depth)]
#         # else:
#         #     raise NotImplementedError
#         # print(self.side_layers)
#
#         block_types = len(route_type) if isinstance(route_type, list) else 1
#
#         self.blocks = nn.ModuleList([
#             VSSBlock(
#                 hidden_dim=side_dim,
#                 T=T,
#                 drop_path=inter_dpr[i],
#                 norm_layer=norm_layer,
#                 channel_first=channel_first,
#                 ssm_d_state=ssm_d_state,
#                 ssm_ratio=ssm_ratio,
#                 ssm_dt_rank=ssm_dt_rank,
#                 ssm_act_layer=ssm_act_layer,
#                 ssm_conv=ssm_conv,
#                 ssm_conv_bias=ssm_conv_bias,
#                 ssm_drop_rate=ssm_drop_rate,
#                 ssm_init=ssm_init,
#                 forward_type=forward_type,
#                 select_type=select_type,
#                 route_type=route_type,
#                 step_size=step_size,
#                 if_noz=if_noz,
#                 if_bidirectional=if_bidirectional,
#                 if_divide_out=if_divide_out,
#                 mlp_ratio=mlp_ratio,
#                 mlp_act_layer=mlp_act_layer,
#                 mlp_drop_rate=mlp_drop_rate,
#                 gmlp=gmlp,
#                 use_checkpoint=use_checkpoint,
#                 _SS=SSBlock,
#             )
#             for i in range(mamba_depth * layer_depth)
#         ])
#         print(f'------------ Total VMamba blocks: {len(self.blocks)}. -------------------')
#         print(f'------------ VMamba d_state: {ssm_d_state}. -------------------')
#
#
#         self.side_linears = nn.ModuleList([nn.Linear(trans_dim, self.side_dim) for _ in range(mamba_depth)])
#         self.side_lns = nn.ModuleList([nn.LayerNorm(trans_dim) for _ in range(mamba_depth)])
#
#         if self.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
#             self.cls_agg = nn.ModuleList(
#                 [ClsAggregator(side_dim, patches=self.patch_embed.grid_size[0], agg_type=self.cls_interaction)for _ in range(mamba_depth)]
#             )
#
#
#         # original init
#         if if_abs_pos_embed:
#             trunc_normal_(self.pos_embed, std=.02)
#         if if_cls_token:
#             trunc_normal_(self.cls_token, std=.02)
#
#         self.apply(self.segm_init_weights)
#
#     # def create_mamba_blocks(self, **kwargs):
#     #     inter_dpr = [x.item() for x in
#     #                  torch.linspace(0, drop_path_rate, mamba_depth * layer_depth)]  # stochastic depth decay rule
#     #     # mamba layers
#     #     # self.side_layers_mode = 'all'  # all, interval, top
#     #     # if self.side_layers_mode == 'all':
#     #     #     self.side_layers = [i for i in range(mamba_depth)]
#     #     # elif self.side_layers_mode == 'interval':
#     #     #     self.side_layers = [i + 1 for i in range(0, mamba_depth, 2)]
#     #     # elif self.side_layers_mode == 'top':
#     #     #     self.side_layers = [i for i in range(mamba_depth // 2, mamba_depth)]
#     #     # else:
#     #     #     raise NotImplementedError
#     #     # print(self.side_layers)
#     #
#     #     block_types = len(route_type) if isinstance(route_type, list) else 1
#     #
#     #     self.blocks = nn.ModuleList([
#     #         VSSBlock(
#     #             hidden_dim=side_dim,
#     #             T=T,
#     #             drop_path=inter_dpr[i],
#     #             norm_layer=norm_layer,
#     #             channel_first=channel_first,
#     #             ssm_d_state=ssm_d_state,
#     #             ssm_ratio=ssm_ratio,
#     #             ssm_dt_rank=ssm_dt_rank,
#     #             ssm_act_layer=ssm_act_layer,
#     #             ssm_conv=ssm_conv,
#     #             ssm_conv_bias=ssm_conv_bias,
#     #             ssm_drop_rate=ssm_drop_rate,
#     #             ssm_init=ssm_init,
#     #             forward_type=forward_type,
#     #             select_type=select_type,
#     #             route_type=route_type,
#     #             step_size=step_size,
#     #             if_noz=if_noz,
#     #             if_bidirectional=if_bidirectional,
#     #             if_divide_out=if_divide_out,
#     #             mlp_ratio=mlp_ratio,
#     #             mlp_act_layer=mlp_act_layer,
#     #             mlp_drop_rate=mlp_drop_rate,
#     #             gmlp=gmlp,
#     #             use_checkpoint=use_checkpoint,
#     #             _SS=SSBlock,
#     #         )
#     #         for i in range(mamba_depth * layer_depth)
#     #     ])
#     #     print(f'------------ Total VMamba blocks: {len(self.blocks)}. -------------------')
#     #     print(f'------------ VMamba d_state: {ssm_d_state}. -------------------')
#
#
#     def segm_init_weights(self, m: nn.Module):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             # NOTE conv was left to pytorch default in my original init
#             lecun_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
#
#
#     def mamba_cls_output(self, hidden_states):
#         if self.if_cls_token:
#             n = hidden_states.shape[-2]
#
#             # 单 CLS token：直接取对应位置
#             idx = self.token_position  # 假设是单个位置（int）
#             cls_token = hidden_states[..., idx, :]  # shape: (B T D)
#             # 生成掩码排除该位置
#             mask = torch.ones(n, dtype=torch.bool, device=hidden_states.device)
#             mask[idx] = False
#
#             # 获取剩下的 hidden states（排除 CLS 位置）
#             remaining = hidden_states[..., mask, :]  # shape: (B T seq_len-N D)（N 是排除的 CLS 数量）
#
#             # 将 CLS 和 remaining 拼接（CLS 在最前面）
#             output = torch.cat([cls_token.unsqueeze(-2), remaining], dim=-2)  # shape: (B T 1+(seq_len-N) D)
#
#             return cls_token, output
#         else:
#             return None, hidden_states
#
#
#
#     def forward(self, image, x_trans_all_hidden, return_hidden=False, inference_params=None):
#         # x_trans_all_hidden : N (B T) L C
#         # Mamba 输入初始化，包括patch embedding，添加cls token，添加 pos embedding，随机打乱 token 序列，以及反转序列
#         x_side = self.patch_embed(rearrange(image, '(b t) c h w -> b c t h w', t=self.T), channel_first=True)
#         x_side = rearrange(x_side, 'b c t l -> (b t) l c').contiguous()
#
#         BT, L, C = x_side.shape   # BT L C
#         trans_depth = len(x_trans_all_hidden)
#
#         ## 为 side mamba 添加cls token， 并将vision transformer 中的cls token 调整到与 mamba 相同的位置
#         if self.if_cls_token:
#             cls_token = self.cls_token.expand(BT, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#             self.token_position = 0
#             x_side = torch.cat((cls_token, x_side), dim=1)
#             L = x_side.shape[1]
#
#         ## pos embedding
#         if self.if_abs_pos_embed:
#             x_side = x_side + self.pos_embed
#             # x_side = self.pos_drop(x_side)
#
#
#         # 送入mamba layer
#         residual = None
#         hidden_states = x_side  # x_side: BT L C
#
#
#         if self.side_layers_mode == 'all':
#             side_layer_index = [i for i in range(trans_depth)]
#         elif self.side_layers_mode == 'interval':
#             side_layer_index = [i + 1 for i in range(0, trans_depth, 2)]
#         elif self.side_layers_mode == 'top':
#             side_layer_index = [i for i in range(trans_depth // 2, trans_depth)]
#         else:
#             raise NotImplementedError
#         # print(side_layer_index)
#
#         x_trans_hidden = x_trans_all_hidden[side_layer_index]
#
#         for i, layer in enumerate(self.blocks):
#             # 将 x_side 和 x_trans 进行拼接
#             xs2xt = self.side_linears[i](self.side_lns[i](x_trans_hidden[i]))
#             hidden_states = hidden_states * 0.5 + xs2xt * 0.5
#
#             x_cls = hidden_states[..., 0, :].unsqueeze(1)  # BT 1 C
#             x = hidden_states[..., 1:, :]
#
#             if self.cls_interaction == 'v1':
#                 ### v1  cls不参与block的计算
#                 x = layer(x, flattened=True)
#
#             elif self.cls_interaction == 'v2':
#                 ### v2  将CLS Token广播到每个空间位置，并与图像特征相加，注入全局信息，再通过全局池化提取CLS信息
#                 x = x + 0.5 * x_cls    # BT L C
#                 x = layer(x, flattened=True)
#                 x_cls_n = torch.mean(x, dim=1, keepdim=True)
#                 x_cls = (x_cls + x_cls_n) / 2.0
#
#
#             elif self.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
#                 ### v3  将CLS Token广播到每个空间位置，再通过空间注意力聚合获得CLS
#
#                 ### v4  将CLS Token广播到每个空间位置，再通过多尺度卷积融合获得CLS
#                 ### 使用不同膨胀率（dilation）的卷积核，捕获不同尺度的上下文信息。
#                 x = x + 0.5 * x_cls    # BT L C
#                 x = layer(x, flattened=True)
#                 # x_cls_n = self.cls_agg(x, flattened=True)
#                 x_cls_n = self.cls_agg[i](x, flattened=True)
#
#                 # cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
#                 #
#                 # cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
#                 #
#                 # x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)
#
#                 x_cls = (x_cls + x_cls_n) / 2.0
#
#             hidden_states = torch.cat([x_cls, x], dim=-2)
#
#         mamba_output = rearrange(hidden_states, '(b t) l dilation -> b t l dilation', t=self.T)
#         mamba_pooled = mamba_output[..., 0, :]
#
#         # mamba_pooled, hidden_states = self.mamba_cls_output(mamba_output)
#
#         return EncoderOutput(
#             pooler_output=mamba_pooled,
#             last_hidden_state=mamba_output
#         )



class SideVMamba(nn.Module):
    def __init__(self,
                 network='vmamba',
                 img_size=224,
                 patch_size=16,
                 channels=3,
                 T=12,
                 stride=16,
                 channel_first=False,
                 # =========================
                 side_layers_mode='all', # all, top, interval
                 mamba_depth=12,
                 layer_depth=1,
                 hierarchical=False,
                 side_dim=192,
                 trans_dim=768,
                 # =========================
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_drop_rate=0.0,
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_init="v0",
                 forward_type="v0_3d",
                 select_type="e",
                 route_type="sy",
                 step_size=-1,
                 if_noz=True,
                 if_bidirectional=True,
                 if_divide_out=True,
                 pos_type='learnable_3d',
                 # =========================
                 norm_epsilon=1e-5,
                 rms_norm=True,
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 gmlp=False,
                 # =========================
                 drop_path_rate=0.1,
                 norm_layer="LN",  # "BN", "LN2D"
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 # device=None,
                 # dtype=None,
                 # if_bidirectional=False,
                 if_abs_pos_embed=True,
                 bimamba_type="",
                 if_cls_token=True,
                 use_checkpoint=False,
                 cls_interaction='v7',
                 **kwargs):
        assert network in ['vmamba', 'vmamba2']
        # factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        # kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mamba_depth = mamba_depth
        self.side_layers_mode = side_layers_mode
        self.hierarchical = hierarchical
        # if hierarchical:
        #     self.hierarchical_weight = nn.Parameter(torch.zeros(4))

        self.if_abs_pos_embed = if_abs_pos_embed

        self.if_cls_token = if_cls_token

        self.num_tokens = 1 if if_cls_token else 0
        self.bimamba_type = bimamba_type
        self.cls_interaction = cls_interaction

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
            bn3d=nn.BatchNorm3d
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        # pretrain parameters
        self.T = T
        self.side_dim = side_dim  # num_features for consistency with other models

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)


        # self.patch_embed = PatchEmbed_3D(
        #         img_size=img_size, patch_size=patch_size, stride=patch_size, in_chans=channels, embed_dim=side_dim)

        # num_patches = self.patch_embed.num_patches


        self.side_pre_bn3d = nn.BatchNorm3d(side_dim)
        self.side_post_ln = nn.LayerNorm(side_dim)
        # Original real-valued video patch embedding:
        self.side_conv1 = conv_3xnxn_std(channels, side_dim, kernel_size=patch_size, stride=patch_size)
        # Use RGB appearance as the real part and temporal differences as the imaginary part.
        # self.side_conv1 = ComplexVideoPatchConv3d(
        #     channels, side_dim, kernel_size=patch_size, stride=patch_size
        # )
        grid_size = (img_size - patch_size) // patch_size + 1
        num_patches = grid_size ** 2

        self.token_position = -1
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.side_dim))

        # if pos_embed == 'learnable_2d':   # 原始的2D位置编码
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.side_dim))
        #
        # elif pos_embed == 'learnable_3d': # 3D 位置编码，可学习
        #     self.pos_embed = nn.Parameter(torch.zeros(1, T, num_patches + self.num_tokens, self.side_dim))
        #
        # elif pos_embed == 'cos_3d': # 3D 余弦位置编码，类似Transformer
        #     self.pos_embed = None

        self.PE = PositionEmbedding(pos_type, self.T, grid_size, grid_size, self.side_dim, self.num_tokens)
        # if if_abs_pos_embed:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.side_dim))

        inter_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, mamba_depth * layer_depth)]  # stochastic depth decay rule

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba layers

        # self.side_layers_mode = 'all'  # all, interval, top
        # if self.side_layers_mode == 'all':
        #     self.side_layers = [i for i in range(mamba_depth)]
        # elif self.side_layers_mode == 'interval':
        #     self.side_layers = [i + 1 for i in range(0, mamba_depth, 2)]
        # elif self.side_layers_mode == 'top':
        #     self.side_layers = [i for i in range(mamba_depth // 2, mamba_depth)]
        # else:
        #     raise NotImplementedError
        # print(self.side_layers)



        # 创建mamba blocks
        param_blocks = [side_dim, T, grid_size, inter_dpr, norm_layer, channel_first, ssm_d_state, ssm_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_init, forward_type, select_type, route_type, step_size, if_noz, if_bidirectional, if_divide_out, mlp_ratio, mlp_act_layer, mlp_drop_rate, gmlp, use_checkpoint, SSBlock, mamba_depth * layer_depth]

        self.blocks = self.create_blocks(*param_blocks)

        # self.blocks = self.create_blocks(
        #     hidden_dim=side_dim,
        #     T=T,
        #     drop_path=inter_dpr,
        #     norm_layer=norm_layer,
        #     channel_first=channel_first,
        #     ssm_d_state=ssm_d_state,
        #     ssm_ratio=ssm_ratio,
        #     ssm_dt_rank=ssm_dt_rank,
        #     ssm_act_layer=ssm_act_layer,
        #     ssm_conv=ssm_conv,
        #     ssm_conv_bias=ssm_conv_bias,
        #     ssm_drop_rate=ssm_drop_rate,
        #     ssm_init=ssm_init,
        #     forward_type=forward_type,
        #     select_type=select_type,
        #     route_type=route_type,
        #     step_size=step_size,
        #     if_noz=if_noz,
        #     if_bidirectional=if_bidirectional,
        #     if_divide_out=if_divide_out,
        #     mlp_ratio=mlp_ratio,
        #     mlp_act_layer=mlp_act_layer,
        #     mlp_drop_rate=mlp_drop_rate,
        #     gmlp=gmlp,
        #     use_checkpoint=use_checkpoint,
        #     _SS=SSBlock,
        #
        # )

        print(f'===========================================================\n'
              f'=== Using {select_type} {route_type} with step_size = {step_size if select_type in ["e", "efficient"] else 1} and bidirectional = {if_bidirectional} ===\n'
              f'===========================================================')
        print(f'------------ Total VMamba blocks: {len(self.blocks)}. -------------------')
        print(f'------------ VMamba d_state: {ssm_d_state}. -------------------')


        self.side_linears = nn.ModuleList([nn.Linear(trans_dim, self.side_dim) for _ in range(mamba_depth)])
        self.side_lns = nn.ModuleList([nn.LayerNorm(trans_dim) for _ in range(mamba_depth)])

        if self.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
            self.cls_agg = nn.ModuleList(
                [ClsAggregator(side_dim, patches=grid_size, agg_type=self.cls_interaction)for _ in range(mamba_depth)]
            )

        # original init
        if if_abs_pos_embed:
            if 'learnable' in pos_type:
                trunc_normal_(self.PE.pos_embed, std=.02)
        if if_cls_token:
            trunc_normal_(self.cls_token, std=.02)

        self.apply(self.segm_init_weights)

    def set_side_layer_index(self, trans_depth):
        '''
        mode: all, interval, top
        找出 side layer 的层数编号，以及不同 route 的最后一次的编号
        '''
        if self.side_layers_mode == 'all':
            assert trans_depth == self.mamba_depth, 'Side layers can not match backbone layers.'
            side_layer_index = [i for i in range(trans_depth)]
        elif self.side_layers_mode == 'interval':
            assert trans_depth % self.mamba_depth == 0, 'Side layers can not match backbone layers.'
            interval = trans_depth // self.mamba_depth
            side_layer_index = [i for i in range(interval - 1, trans_depth, interval)]
        elif self.side_layers_mode == 'top':
            assert trans_depth >= self.mamba_depth, 'Side layers can not match backbone layers.'
            side_layer_index = [i for i in range(trans_depth - self.mamba_depth, trans_depth)]
        else:
            raise NotImplementedError

        route_interval = int(len(side_layer_index) // 4)

        side_layer_route_index = side_layer_index[route_interval-1::route_interval]

        print(f'Side Layer Index: {side_layer_index}')
        print(f'Side Layer Route Index: {side_layer_route_index}')
        return side_layer_index, side_layer_route_index

    def create_blocks(self, hidden_dim, T, grid_size, drop_path, norm_layer, channel_first, ssm_d_state, ssm_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_init, forward_type, select_type, route_type, step_size, if_noz, if_bidirectional, if_divide_out,  mlp_ratio, mlp_act_layer, mlp_drop_rate, gmlp, use_checkpoint, _SS, num_blocks):

        assert isinstance(route_type, list), 'route_type must be a list'
        num_route_types = len(route_type)
        assert num_blocks % num_route_types == 0
        route_type_list = []
        blocks_per_type = num_blocks // num_route_types

        #  r1
        for t in route_type:
            route_type_list.extend([t] * blocks_per_type)

        ##  r2
        # for i in range(blocks_per_type):
        #     route_type_list.extend(route_type)


        assert len(route_type_list) == num_blocks, f'{len(route_type_list)} route types must have {num_blocks} blocks'
        print(f'Total number of blocks: {num_blocks}, divided into {len(route_type)} route types, final route list: {route_type_list}')


        # 创建mamba blocks
        blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=hidden_dim,
                T=T,
                grid_size=grid_size,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                select_type=select_type,
                route_type=route_type_list[i],
                step_size=step_size,
                if_noz=if_noz,
                if_bidirectional=if_bidirectional,
                if_divide_out=if_divide_out,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS=SSBlock,
            )
            for i in range(num_blocks)
        ])

        return blocks


    def segm_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}


    def mamba_cls_output(self, hidden_states):
        if self.if_cls_token:
            n = hidden_states.shape[-2]

            # 单 CLS token：直接取对应位置
            idx = self.token_position  # 假设是单个位置（int）
            cls_token = hidden_states[..., idx, :]  # shape: (B T D)
            # 生成掩码排除该位置
            mask = torch.ones(n, dtype=torch.bool, device=hidden_states.device)
            mask[idx] = False

            # 获取剩下的 hidden states（排除 CLS 位置）
            remaining = hidden_states[..., mask, :]  # shape: (B T seq_len-N D)（N 是排除的 CLS 数量）

            # 将 CLS 和 remaining 拼接（CLS 在最前面）
            output = torch.cat([cls_token.unsqueeze(-2), remaining], dim=-2)  # shape: (B T 1+(seq_len-N) D)

            return cls_token, output
        else:
            return None, hidden_states



    # def forward(self, image, x_trans_all_hidden, return_hidden=False, inference_params=None):
    #     '''
    #     不用
    #     '''
    #     # x_trans_all_hidden : N (B T) L C
    #     # Mamba 输入初始化，包括patch embedding，添加cls token，添加 pos embedding，随机打乱 token 序列，以及反转序列
    #     x_side = self.patch_embed(rearrange(image, '(b t) c h w -> b c t h w', t=self.T), channel_first=True)
    #     x_side = rearrange(x_side, 'b c t l -> (b t) l c').contiguous()
    #
    #     BT, L, C = x_side.shape   # BT L C
    #     trans_depth = len(x_trans_all_hidden)
    #
    #     ## 为 side mamba 添加cls token， 并将vision transformer 中的cls token 调整到与 mamba 相同的位置
    #     if self.if_cls_token:
    #         cls_token = self.cls_token.expand(BT, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #         self.token_position = 0
    #         x_side = torch.cat((cls_token, x_side), dim=1)
    #         L = x_side.shape[1]
    #
    #     ## pos embedding
    #     if self.if_abs_pos_embed:
    #         x_side = self.PE.add_pos_embed(x_side)
    #
    #         # x_side = rearrange(x_side, '(b t) l c -> b t l c', t=self.T).contiguous()
    #         # x_side = x_side + self.PE.pos_embed
    #         # x_side = rearrange(x_side, 'b t l c -> (b t) l c').contiguous()
    #         # x_side = self.pos_drop(x_side)
    #
    #
    #     # 送入mamba layer
    #     residual = None
    #     hidden_states = x_side  # x_side: BT L C
    #
    #
    #     if self.side_layers_mode == 'all':
    #         side_layer_index = [i for i in range(trans_depth)]
    #     elif self.side_layers_mode == 'interval':
    #         side_layer_index = [i + 1 for i in range(0, trans_depth, 2)]
    #     elif self.side_layers_mode == 'top':
    #         side_layer_index = [i for i in range(trans_depth // 2, trans_depth)]
    #     else:
    #         raise NotImplementedError
    #     # print(side_layer_index)
    #
    #     x_trans_hidden = x_trans_all_hidden[side_layer_index]
    #
    #     for i, layer in enumerate(self.blocks):
    #         # 将 x_side 和 x_trans 进行拼接
    #         xs2xt = self.side_linears[i](self.side_lns[i](x_trans_hidden[i]))
    #         hidden_states = hidden_states * 0.5 + xs2xt * 0.5
    #
    #         x_cls = hidden_states[..., 0, :].unsqueeze(1)  # BT 1 C
    #         x = hidden_states[..., 1:, :]
    #
    #         if self.cls_interaction == 'v1':
    #             ### v1  cls不参与block的计算
    #             x = layer(x, flattened=True)
    #
    #         elif self.cls_interaction == 'v2':
    #             ### v2  将CLS Token广播到每个空间位置，并与图像特征相加，注入全局信息，再通过全局池化提取CLS信息
    #             x = x + 0.5 * x_cls    # BT L C
    #             x = layer(x, flattened=True)
    #             x_cls_n = torch.mean(x, dim=1, keepdim=True)
    #             x_cls = (x_cls + x_cls_n) / 2.0
    #
    #
    #         elif self.cls_interaction in ['v3', 'v4', 'v5', 'v6', 'v7', 'v8']:
    #             ### v3  将CLS Token广播到每个空间位置，再通过空间注意力聚合获得CLS
    #
    #             ### v4  将CLS Token广播到每个空间位置，再通过多尺度卷积融合获得CLS
    #             ### 使用不同膨胀率（dilation）的卷积核，捕获不同尺度的上下文信息。
    #             x = x + 0.5 * x_cls    # BT L C
    #             x = layer(x, flattened=True)
    #             # x_cls_n = self.cls_agg(x, flattened=True)
    #             x_cls_n = self.cls_agg[i](x, flattened=True)
    #
    #             # cls_weight = self.cls_weight_fc(x).squeeze()  # BT L C  ->  BT L
    #             #
    #             # cls_weight = torch.softmax(cls_weight, dim=-1)  # BT L
    #             #
    #             # x_cls_n = torch.einsum('blc,bl->bc', [x, cls_weight]).unsqueeze(1)
    #
    #             x_cls = (x_cls + x_cls_n) / 2.0
    #
    #         hidden_states = torch.cat([x_cls, x], dim=-2)
    #
    #     mamba_output = rearrange(hidden_states, '(b t) l dilation -> b t l dilation', t=self.T)
    #     mamba_pooled = mamba_output[..., 0, :]
    #
    #     # mamba_pooled, hidden_states = self.mamba_cls_output(mamba_output)
    #
    #     return EncoderOutput(
    #         pooler_output=mamba_pooled,
    #         last_hidden_state=mamba_output
    #     )


if __name__ == "__main__":
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    def test3d():
        image = torch.zeros([1, 6, 4, 4, 3]).to(device)
        b, t, h, w, c = image.shape

        image = image.view(b, -1, c)
        for i in range(t * h * w):
            image[:, i, :] = i + 1
        image = image.view(b, t, h, w, c)


        model = SSBlock(image.shape[-1],
                        forward_type="t1_3d",
                        ssm_ratio=1.0,
                        select_type="s",
                        route_type="sy",
                        step_size=2,
                        if_noz=True,
                        if_bidirectional=True,
                        if_divide_out=True,).to(device)
        model(image, flattened=False)


    def test2d():
        image = torch.zeros([1, 14, 14, 192]).to(device)
        b, h, w, c = image.shape

        image = image.view(b, -1, c)
        for i in range(h * w):
            image[:, i, :] = i
        image = image.view(b, h, w, c)

        model = SSBlock(image.shape[-1], forward_type="t0_noz", ssm_ratio=1.0).to(device)
        model(image, flattened=False)

    test3d()
