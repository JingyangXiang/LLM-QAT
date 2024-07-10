import gc
import typing

import numpy as np
import torch
import torch.nn as nn
import tqdm


class RotateModule(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, dtype=torch.float32, matrix_cost='min'):
        super(RotateModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dtype = dtype
        self.matrix_cost = matrix_cost
        self.register_rotate_matrix(matrix_cost=matrix_cost)

    def register_rotate_matrix(self, matrix_cost='min'):
        assert matrix_cost in ['min', 'max']
        assert matrix_cost == 'min', 'only support min now'
        N1, N2 = get_greatest_common_factor(self.hidden_size // self.num_attention_heads)
        assert int(N1 * N2) == self.hidden_size // self.num_attention_heads
        self.register_parameter("R1", nn.Parameter(get_orthogonal_matrix(N1, mode='random', dtype=self.dtype)))
        self.register_parameter("R2", nn.Parameter(get_orthogonal_matrix(N2, mode='random', dtype=self.dtype)))

    def forward(self):
        return torch.kron(self.R1, self.R2)


@torch.no_grad()
def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """fuse the linear operations in Layernorm into the adjacent linear blocks."""
    # 将RMSNorm/LayerNorm的权重fuse到下一层的输入通道上
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype).clone()

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            raise NotImplementedError

    layernorm.weight.data = torch.ones_like(layernorm.weight).to(layernorm.weight.dtype)


@torch.no_grad()
def fuse_layer_norms(model):
    layers = model.model.layers

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # 每个QKV在输入前会有一个post_attention_layernorm
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        # 每个LlamaDecoderLayer的AttentionLayer
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])

    fuse_ln_linear(model.model.norm, [model.lm_head])


def get_greatest_common_factor(N):
    # 获取最大公因数
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i


@torch.inference_mode()
def init_rotate_to_model(model):
    Q = RotateModule(model.config.hidden_size)

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    # 给最初的模型应用, 输入的特征, 给embedding的输出用的
    # 在输入lm_head之前需要把特征旋转回来
    model.model.Q = Q
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers

    # 刚开始的特征需要旋转, 旋转从Embedding开始
    model.model.RotateEmbedding = Q

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        """这些都是可以offline计算的旋转矩阵, 受到结构的限制, 这里全局都是一个Q"""
        # QKV的权重需要乘一个旋转矩阵进行量化
        layer.self_attn.q_proj.RotateWeightIn = Q
        layer.self_attn.k_proj.RotateWeightIn = Q
        layer.self_attn.v_proj.RotateWeightIn = Q

        # Attention output的输出维度的旋转
        layer.self_attn.o_proj.RotateWeightOut = Q

        # FFN层的up_proj和gate_proj
        layer.mlp.up_proj.RotateWeightIn = Q
        layer.mlp.gate_proj.RotateWeightIn = Q

        # FFN的输出
        layer.mlp.down_proj.RotateWeightOut = Q

        """剩下的都是需要Online计算的Rotate Matrix"""
        # KVCache的旋转塞给Attention

        # KVCache可以直接旋转, 一个转Q, 一个转Q.t()就好
        layer.self_attn.RotateKV = nn.Identity()
        # 由于只有这里要online, 因此选择在Linear层里面只旋转Weight
        # feature的旋转都放在外部显式的做, 但是feature的量化放到Linear里面去做
        layer.mlp.RotateGate = nn.Identity()
        # 这里要对应的再去旋转一下down_proj, 因为外部的x已经通过旋转
        layer.mlp.down_proj.RotateWeightIn = nn.Identity()

    """经过这些操作之后, 在保持教师和学生都是FP32的情况下, 第一步教师和学生的输出应该是相等的"""
    return Q


def get_orthogonal_matrix(size, mode, dtype=torch.float32):
    if mode == 'random':
        return random_orthogonal_matrix(size, dtype)
    else:
        raise ValueError(f'Unknown mode {mode}')


def random_orthogonal_matrix(size, dtype=torch.float32):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=dtype)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q
