import torch
import torch.nn as nn
import math
from einops import rearrange


def rotate_half(x):
    """旋转操作：将向量的前半部分和后半部分进行旋转"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def broadcat(tensors, dim=-1):
    """广播拼接函数"""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim

    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims[dim]) if val == 1]
    assert len(expandable_dims) > 0, 'invalid operation, no single dimensions to expand across'

    for i, _ in expandable_dims:
        shapes = [(t.shape[:dim] + (t.shape[dim] if t.shape[dim] != 1 else 0,) + t.shape[dim + 1:])
                  for t in tensors]
        max_len = max([shape[i] for shape in shapes if i < len(shape)])
        for j, t in enumerate(tensors):
            if t.shape[i] == 1:
                t = t.expand(*t.shape[:i], max_len, *t.shape[i + 1:])
                tensors[j] = t

    return torch.cat(tensors, dim=dim)


def repeat(tensor, pattern):
    """简单的重复模式模拟"""
    if '(n r)' in pattern and 'r' in pattern:
        # 假设模式是 "... n -> ... (n r)"，其中 r 是重复次数
        original_shape = tensor.shape
        new_shape = list(original_shape[:-1]) + [original_shape[-1] * 2]  # r=2
        result = tensor.repeat(1, 1, 2) if len(tensor.shape) == 3 else tensor.repeat(1, 2)
        return result
    return tensor


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
            self,
            dim,
            pt_seq_len=16,  # 预训练序列长度
            ft_seq_len=None,  # 微调序列长度
            custom_freqs=None,
            freqs_for='lang',  # 频率类型
            theta=10000,
            max_freq=10,
            num_freqs=1,
    ):
        super().__init__()

        print("=== VisionRotaryEmbeddingFast 初始化 ===")
        print(f"参数: dim={dim}, pt_seq_len={pt_seq_len}, ft_seq_len={ft_seq_len}")

        if custom_freqs:
            freqs = custom_freqs
            print("使用自定义频率")
        elif freqs_for == 'lang':
            # 语言模型的标准 RoPE 频率
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
            print("使用语言模型频率")
        elif freqs_for == 'pixel':
            # 像素级别频率（用于视觉）
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
            print("使用像素级别频率")
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
            print("使用常数频率")
        else:
            raise ValueError(f'未知的模态 {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        print(f"微调序列长度: {ft_seq_len}, 预训练序列长度: {pt_seq_len}")

        # 创建位置索引并进行长度归一化
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        print(f"位置索引 t: {t[:5]}... (共{len(t)}个)")

        # 计算频率
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        print(f"频率张量形状: {freqs.shape}")

        # 将频率重复两次以匹配维度
        freqs = repeat(freqs, '... n -> ... (n r)')
        print(f"重复后频率形状: {freqs.shape}")

        # 创建二维空间的频率（通过广播拼接）
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        print(f"广播拼接后频率形状: {freqs.shape}")

        # 计算余弦和正弦值
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        # 注册为缓冲区（不参与梯度更新）
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print('======== RoPE 频率形状 ========')
        print(f'freqs_cos shape: {self.freqs_cos.shape}')
        print(f'freqs_sin shape: {self.freqs_sin.shape}')
        print('================================')

    def forward(self, t):
        """
        应用旋转位置编码
        t: 输入张量 [batch, seq_len, dim]
        """
        print(f"输入张量形状: {t.shape}")

        if t.shape[1] % 2 != 0:
            # 如果序列长度为奇数，特殊处理
            print("检测到奇数长度序列，特殊处理...")
            t_spatial = t[:, 1:, :]  # 排除第一个元素（可能是类别token）

            # 应用 RoPE: x*cos + rotate_half(x)*sin
            rotated_t = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin

            # 重新组合（保留第一个元素）
            result = torch.cat((t[:, :1, :], rotated_t), dim=1)
            return result
        else:
            # 序列长度为偶数，直接应用 RoPE
            print("偶数长度序列，直接应用 RoPE")
            result = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
            return result

class PositionEmbedding(nn.Module):
    def __init__(self, pos_type, t_len, h_len, w_len, d_model, num_tokens):
        super().__init__()
        self.pos_type = pos_type
        self.t, self.h, self.w, self.d = t_len, h_len, w_len, d_model

        if pos_type == 'learnable_2d':   # 原始的2D位置编码
            self.pos_embed = nn.Parameter(torch.zeros([1, h_len * w_len + num_tokens, d_model]))

        elif pos_type == 'learnable_3d': # 3D 位置编码，可学习
            self.pos_embed = nn.Parameter(torch.zeros([1, t_len, h_len * w_len + num_tokens, d_model]))

        elif pos_type == 'cos_3d': # 3D 余弦位置编码，类似Transformer
            pos_embed = self.positional_encoding_3d(1, t_len, h_len, w_len, d_model)
            # 注册为缓冲区（不参与梯度更新）
            self.register_buffer("pos_embed", pos_embed)

        else:
            raise ValueError(f'Unknown type {pos_type}')

    def add_pos_embed(self, x):
        if self.pos_type == 'learnable_2d':
            return x + self.pos_embed

        elif self.pos_type == 'learnable_3d':
            x = rearrange(x, '(b t) l c -> b t l c', t=self.t).contiguous()
            return rearrange(x + self.pos_embed, 'b t l c -> (b t) l c').contiguous()

        elif self.pos_type == 'cos_3d':
            x = rearrange(x, '(b t) l c -> b t l c', t=self.t).contiguous()
            return rearrange(x + self.pos_embed, 'b t l c -> (b t) l c').contiguous()
        else:
            raise ValueError(f'Unknown type {x.pos_type}')

    def positional_encoding_1d(self, seq_len, d_model):
        """
        生成正弦/余弦一维位置编码。

        Args:
            seq_len (int): 序列长度（如 token 数量）
            d_model (int): 嵌入维度

        Returns:
            torch.Tensor: 形状为 (1, seq_len, d_model) 的位置编码
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

        # 计算 div_term = 1 / (10000^(2i / d_model)) = exp(-ln(10000) * 2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model // 2,)

        # 偶数列：sin, 奇数列：cos
        pe[:, 0::2] = torch.sin(position * div_term)      # 所有行，偶数列
        pe[:, 1::2] = torch.cos(position * div_term)      # 所有行，奇数列

        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def positional_encoding_3d(self, batch_size, t_len, h_len, w_len, d_model):
        # 总长度 = t_len * h_len * w_len
        # pe = torch.zeros(t_len, h_len, w_len, d_model)

        pe_t = self.positional_encoding_1d(t_len, d_model)  # (1, T, d)
        pe_h = self.positional_encoding_1d(h_len, d_model)  # (1, H, d)
        pe_w = self.positional_encoding_1d(w_len, d_model)  # (1, W, d)

        # 扩展维度以便广播相加
        pe_t = pe_t.view(1, t_len, 1, 1, d_model)
        pe_h = pe_h.view(1, 1, h_len, 1, d_model)
        pe_w = pe_w.view(1, 1, 1, w_len, d_model)

        pe_3d = pe_t + pe_h + pe_w  # (1, T, H, W, d_model)
        pe_3d = pe_3d.flatten(start_dim=2, end_dim=3) # (1, T, H * W, d_model)
        cls_pos_emb = nn.Parameter(torch.zeros([1,t_len,1,d_model]))
        pe_3d = torch.cat((cls_pos_emb, pe_3d), dim=2)
        return pe_3d  # (1, T, H * W + 1, d_model)

def test_RoPE():
    print("=== 测试 VisionRotaryEmbeddingFast ===")
    # 创建测试输入
    batch_size = 2
    seq_len = 25  # 奇数长度测试
    embed_dim = 128

    img_size = 224
    patch_size = 16

    half_head_dim = embed_dim // 2
    hw_seq_len = img_size // patch_size
    # 创建实例
    rotary_emb = VisionRotaryEmbeddingFast(
        dim=half_head_dim,
        pt_seq_len=16,
        ft_seq_len=hw_seq_len,  # 微调时使用更长的序列
        freqs_for='pixel'  # 用于视觉任务
    )

    test_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\n测试输入形状: {test_input.shape}")

    # 应用旋转编码
    output = rotary_emb(test_input)
    print(f"输出形状: {output.shape}")

    print("\n=== 功能总结 ===")
    print("1. VisionRotaryEmbeddingFast 是一个视觉旋转位置编码类")
    print("2. 支持预训练和微调时不同的序列长度（长度外推）")
    print("3. 为视觉任务生成2D空间位置的旋转编码")
    print("4. 支持奇偶数长度序列的不同处理方式")
    print("5. 使用 RoPE 公式: x*cos + rotate_half(x)*sin")

def test_PE():
    print("测试 PositionEmbedding")
    pos_type = 'cos_3d'
    shape = [1, 12, 7, 7, 192]
    PE = PositionEmbedding(pos_type=pos_type, shape=shape)

    print(PE.pos_embed.shape)



# 测试代码
if __name__ == "__main__":
    test_PE()