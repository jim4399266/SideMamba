from torchmetrics import Metric

from typing import Any, Union, Optional, List, Dict
import logging
import numpy as np
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

from typing import Any, ContextManager, List, Tuple
from collections import OrderedDict, UserDict
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


@dataclass
class EncoderOutput(ModelOutput):
    """
        Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
        Args:
            pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
                CLS token or pooled hidden-states at the output of the last layer of the model.

            last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.

                If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
                hidden_size)` is output.

            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

        """
    pooler_output: Union[torch.FloatTensor, None] = None
    last_hidden_state: Union[torch.FloatTensor, None] = None
    hidden_states: Union[Tuple[torch.FloatTensor], torch.FloatTensor, torch.Tensor, None] = None

@dataclass
class EvalCacheOutput(ModelOutput):
    """
        Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
        Args:
            pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
                CLS token or pooled hidden-states at the output of the last layer of the model.

            last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.

                If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
                hidden_size)` is output.

            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

        """
    index: Union[torch.LongTensor, torch.Tensor]= None
    mask: Union[torch.LongTensor, torch.Tensor] = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Union[Tuple[torch.FloatTensor], torch.FloatTensor, torch.Tensor] = None

@dataclass
class EvalReorderedOutput(ModelOutput):

    mask: Union[List[torch.LongTensor], List[torch.Tensor]] = None
    pooler_output: Union[List[torch.FloatTensor], List[torch.Tensor]] = None
    last_hidden_state: Union[List[torch.FloatTensor], List[torch.Tensor]] = None
    hidden_states: Union[Tuple[torch.FloatTensor], List[torch.FloatTensor], torch.FloatTensor, torch.Tensor] = None


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Accuracy(Metric):
    def __init__(self, ignore_id=-100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.ignore_id = ignore_id

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device)
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != self.ignore_id]
        target = target[target != self.ignore_id]
        if target.numel() == 0:
            return 1
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('scalar', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar, dtype=torch.float).to(self.scalar.device)
        self.scalar += scalar
        self.total += 1
        # print('\n', self.scalar)

    def compute(self):
        return self.scalar / self.total

class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total

##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class NCELoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using NCE Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = len(prediction)
        probs1 = F.log_softmax(prediction, 1)   # bs bs
        probs2 = F.softmax(label * 10, 1)       #
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()




class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class LayerNorm_conv(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor):
        x = x.permute(0,2,3,1)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))# add ssf
        return ret.type(orig_type).permute(0,3,1,2)




class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_first=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_first = channels_first

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_first:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

    @classmethod
    def convert_ln_to_dyt(cls, module, module_name=None):
        module_output = module
        if isinstance(module, nn.LayerNorm):
            # print(f'convert module name: {module_name}')
            module_output = cls(module.normalized_shape, not isinstance(module, LayerNorm2d))

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_ln_to_dyt(child, name))
        del module
        return module_output


class DyTConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, shape,padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, patches=14, padding=None, groups=1, dilation=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()


        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)

        self.DyT = DynamicTanh([out_channels, patches, patches] )    # channels first
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def autopad(self, kernel_size, padding=None, dilation=1):  # kernel, padding, dilation
        """Pad to 'same' shape outputs."""
        if dilation > 1:
            kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for x in kernel_size]  # actual kernel-size
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
        return padding

    def forward(self, x):
        """Apply convolution, DynamicTanh  and activation to input tensor."""
        x = self.conv(x)
        out = self.bn(x*self.DyT(x))  #大家可以合理玩一下这个DyT模块，但是不要直接替换bn批标准化，不然容易造成训练不稳定。
        return self.act(out)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Scaler(nn.Module):
    def __init__(self, dim, scale_factors=None, scaling_type='v1'):
        super(Scaler, self).__init__()
        assert scaling_type in ['v1', 'v2']
        if scale_factors is None:
            if scaling_type == 'v1':
                scale_factors = [0.25, 0.5, 1.0]  # feature scales used
            else:
                scale_factors = [0.5, 2.0, 4.0]
        self.dim = dim
        self.scaling_type = scaling_type

        self.stages = nn.ModuleList([self.create_scaler_block_s(f) for f in scale_factors])
        # self.stages = nn.ModuleList([self.create_scaler_block(f) for f in scale_factors])


    def create_scaler_block_s(self, scale_factor):
        dim = self.dim
        if scale_factor == 2.0:
            layers = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        elif scale_factor == 1.0:
            layers = nn.Identity()
        elif scale_factor == 0.5:
            layers = nn.MaxPool2d(kernel_size=2, stride=2)
        elif scale_factor == 0.25:
            layers = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported yet.")
        return layers


    def create_scaler_block(self, scale_factor):
        dim, out_dim, out_channels = self.dim, self.dim, self.dim

        if scale_factor == 4.0:
            layers = [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                LayerNorm_conv(dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            ]
            out_dim = dim // 4
        elif scale_factor == 2.0:
            layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
            out_dim = dim // 2
        elif scale_factor == 1.0:
            layers = []
        elif scale_factor == 0.5:
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif scale_factor == 0.25:
            layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported yet.")

        layers.extend(
            [
                nn.Conv2d(
                    out_dim,
                    out_channels,
                    kernel_size=1,
                ),
                LayerNorm_conv(out_channels),
                nn.GELU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                LayerNorm_conv(out_channels)
            ]
        )
        layers = nn.Sequential(*layers)
        return layers


    def forward(self, input, stages):
        B, T, L, C = input.shape
        H = W = int(math.sqrt(L - 1))

        offset = []

        cls = input[..., 0, :]
        offset.append(cls.shape[1])

        hidden_state = input[..., 1:, :].reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        scaler_ms = []
        for stage in stages:
            tmp = stage(hidden_state)
            tmp = tmp.view(B, T, C, -1).permute(0, 1, 3, 2)
            tmp = tmp.view(B, -1, C)
            offset.append(offset[-1] + tmp.shape[1])
            scaler_ms.append(tmp)

        # scaler_ms = [vi.view(B, -1, C) for vi in scaler_ms]
        scaler_st = torch.cat(scaler_ms, dim=1)
        scaler_output = torch.cat((cls, scaler_st), dim=1)

        return scaler_output, offset





def statistic(model, *args):
    from thop import profile
    from torchinfo import summary
    # 统计计算量的时候，最好 batch size 统一设置为 1，因为 batch size 会影响计算量

    # 1. 使用 torchinfo 计算 MACs (Multiply-Accumulate Operations) 和 参数量
    print("Torchinfo for Params and FLOPs")
    # 直接打印详细报告，包含每层的参数和 Mult-Adds
    model_stats = summary(model, input_data=(*args,), verbose=1)

    # 通常 1 MAC ≈ 2 FLOPs
    flops = model_stats.total_mult_adds * 2
    print(f"参数量 (Params): {model_stats.total_params / 1e6:.2f} M (百万)")
    print(f"可训练参数量 (Native): {model_stats.trainable_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据

    print(f"计算量 (GFLOPs): {flops / 1e9:.2f} G (十亿)")
    print(f"计算量 (TFLOPs): {flops / 1e12:.4f} T")

    # 2. 使用 thop 计算 MACs (Multiply-Accumulate Operations) 和 参数量
    print("Thop for Params and FLOPs")
    macs, params = profile(model, inputs=(*args,), verbose=False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_side_params = sum(p.numel() for n,p in model.named_parameters() if (p.requires_grad and 'side' in n))

    # 通常 1 MAC ≈ 2 FLOPs
    flops = macs * 2
    print(f"参数量 (Params): {params / 1e6:.2f} M (百万)")
    print(f"可训练参数量 (Native): {trainable_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据
    print(f"可训练侧网络参数量 (Side): {trainable_side_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据

    print(f"计算量 (GFLOPs): {flops / 1e9:.2f} G (十亿)")
    print(f"计算量 (TFLOPs): {flops / 1e12:.4f} T")

    ...
    # exit()


# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    def test1():
        input = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
        DyT = DynamicTanh([32,128,128])
        output = DyT(input)  # 通过 DyTConv 模块计算输出
        print('DyT_Input size:', input.size())  # 打印输入张量的形状
        print('DyT_Output size:', output.size())  # 打印输出张量的形状


        input_tensor = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
        # 创建 DyTConv 模块实例，输入通道数为32，输出通道数为 64，卷积核为1，步长为1。
        # module =DyTConv(32,64,1,1,[64,128,128])
        module =DyTConv(32,64,3,2,64)
        output_tensor = module(input_tensor)  # 通过 DyTConv 模块计算输出
        print('DyTConv_Input size:', input_tensor.size())  # 打印输入张量的形状
        print('DyTConv_Output size:', output_tensor.size())  # 打印输出张量的形状

    def test2():
        input = torch.ones(12, 3, 14, 14)  # 创建一个形状为 (1,32,128, 128)
        DyT = DynamicTanh([3, 14, 14])
        output = DyT(input)  # 通过 DyTConv 模块计算输出
        print('DyT_Input size:', input.size())  # 打印输入张量的形状
        print('DyT_Output size:', output.size())  # 打印输出张量的形状

        input_tensor = torch.ones(12, 3, 14, 14)  # 创建一个形状为 (1,32,128, 128)
        # 创建 DyTConv 模块实例，输入通道数为32，输出通道数为 64，卷积核为1，步长为1。
        # module =DyTConv(32,64,1,1,[64,128,128])
        module = DyTConv(3, 6, 3, 1, input_tensor.shape[-1])
        output_tensor = module(input_tensor)  # 通过 DyTConv 模块计算输出
        print('DyTConv_Input size:', input_tensor.size())  # 打印输入张量的形状
        print('DyTConv_Output size:', output_tensor.size())  # 打印输出张量的形状

    test2()