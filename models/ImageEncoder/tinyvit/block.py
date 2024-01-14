import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath as TimmDropPath

from .utils import Conv2d_BN, DropPath, Mlp


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            # self.register_buffer('ab',
            #                    self.attention_biases[:, self.attention_bias_idxs],
            #                    persistent=False)
    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x

class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, args, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"