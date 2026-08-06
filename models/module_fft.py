import torch
import torch.nn as nn

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class FFParser_n(nn.Module):
    def __init__(self, dim, h=128, w=239, t=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, t, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, T, H, W = x.shape
        assert H == W, "height and width are not equal"

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfftn(x, s=(T, H, W), dim=(2, 3, 4), norm='ortho')

        x = x.reshape(B, C, T, H, W)

        return x

class VideoFFParser(nn.Module):
    def __init__(self, C=192, T=12, H=7, W=7):
        super().__init__()
        self.C, self.T, self.H, self.W = C, T, H, W
        # rfftn on (T, H, W) → freq shape: (T, H, W//2 + 1) = (12, 7, 4)
        freq_shape = (C, T, H, W // 2 + 1)  # (192, 12, 7, 4)
        self.complex_weight = nn.Parameter(
            torch.randn(*freq_shape, 2) * 0.02  # last dim: (real, imag)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert C == self.C and (T, H, W) == (self.T, self.H, self.W)
        ori_dtype = x.dtype
        # FFT
        # x_perm = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        x_fft = torch.fft.rfftn(x.to(torch.float32), dim=(-3, -2, -1), norm='ortho')  # [B, C, 4, 7, 4]

        # Apply learnable filter
        weight = torch.view_as_complex(self.complex_weight)  # [C, 4, 7, 4]
        x_filtered = x_fft * weight.unsqueeze(0)  # broadcast B

        # IFFT
        x_out = torch.fft.irfftn(x_filtered, s=(T, H, W), dim=(-3, -2, -1), norm='ortho')
        # x_out = x_out.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]

        return x_out.to(ori_dtype) + x  # residual connection (optional)

class Spectral_Layer(nn.Module):
    def __init__(self, t, h, w, dim):
        super().__init__()
        self.dim = dim
        self.t = t
        self.h = h
        self.w = w

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim//2)
        # self.ffp_module = FFParser_n(dim, h=self.h, w=self.w, t=self.t)
        self.ffp_module = VideoFFParser(dim, self.t, self.h, self.w)

    def forward(self, x, channel_first=True):
        if not channel_first:
            x = x.permute(0, 4, 1, 2, 3)    # b, t, h, w, c -> b, c, t, h, w

        B, C = x.shape[:2]
        # B, C, DIM1, DIM2, DIM3
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # print(x.shape,'shape')

        x_reshape = x.reshape(B, C, n_tokens).transpose(-1, -2)
        norm1_x = self.norm1(x_reshape)
        norm1_x = norm1_x.reshape(B, C, *img_dims)
        x_fft = self.ffp_module(norm1_x)
        # print(x_fft.shape, 'xfft')
        norm2_x_fft = self.norm2(x_fft.reshape(B, C, n_tokens).transpose(-1, -2))
        x_spatial = self.mlp(norm2_x_fft.transpose(-1, -2).reshape(B, C, *img_dims))
        out_all = x + x_spatial
        new_out = out_all.transpose(-1, -2).reshape(B, C, *img_dims)

        if not channel_first:
            new_out = new_out.permute(0, 2, 3, 4, 1)    # b, c, t, h, w -> b, t, h, w, c
        return new_out

if __name__ == "__main__":
    print("=== 测试 FFT ===")
    # 创建测试输入
    batch_size = 2
    T = 4
    h = 7
    w = 7
    # seq_len = 50  # 奇数长度测试
    embed_dim = 192

    image_tensor = nn.Parameter(torch.randn(batch_size, T, h, w, embed_dim))

    block = Spectral_Layer(T, h, w, embed_dim)

    a = block(image_tensor, channel_first=False)

    print(a.shape)

