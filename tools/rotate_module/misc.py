from torch import nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = LlamaRMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.scale

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if hasattr(self, 'qk_rotate'):
            q = self.qk_rotate(q, mode='data_qk')
            k = self.qk_rotate(k, mode='data_qk')
            print("rotate")

        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_cls = self.proj(x_cls) + x

        return x_cls


class FFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.act_fn = nn.SiLU()
        self.norm = LlamaRMSNorm(hidden_size)

    def forward(self, x):
        x = self.norm(x)
        output = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if hasattr(self, 'RotateDataIn'):
            output = self.RotateDataIn(output, mode='data_input')
            print("online part")
        return self.down_proj(output) + x
