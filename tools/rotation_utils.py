import gc
import math
import typing

import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import einsum, rearrange
from scipy.linalg import hadamard
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class RotateModule(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, mode='random', dtype=torch.float32, matrix_cost='min'):
        super(RotateModule, self).__init__()
        self.params_dict = None
        # 定义分解多少个矩阵
        self.len_R = 0
        # 定义因数
        self.Ns = None
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dtype = dtype
        self.matrix_cost = matrix_cost
        self.mode = mode
        self.register_rotate_matrix(matrix_cost=matrix_cost)

    def register_rotate_matrix(self, matrix_cost='min'):
        assert matrix_cost in ['min', 'max']
        assert matrix_cost == 'min', 'only support min now'
        self.Ns = get_greatest_common_factor(self.hidden_size // self.num_attention_heads)
        self.len_R = len(self.Ns)
        assert np.prod(self.Ns) == self.hidden_size // self.num_attention_heads
        assert len(self.Ns) == 2, f"only support len(Ns) = {len(self.Ns)} now"

        param_dict = {}

        for index, N in enumerate(self.Ns):
            if self.num_attention_heads == 1:
                param_dict[f'R{index}'] = nn.Parameter(get_orthogonal_matrix(N, mode=self.mode, dtype=self.dtype))
            else:
                # for i in range(self.num_attention_heads):
                # param_dict[f'R{index}'] = nn.Parameter(
                #     repeat(get_orthogonal_matrix(N, mode='random', dtype=self.dtype), 'i j -> k i j',
                #            k=self.num_attention_heads))
                # 产生self.num_attention_heads个不同的矩阵
                param_dict[f'R{index}'] = nn.Parameter(
                    torch.stack([get_orthogonal_matrix(N, mode=self.mode, dtype=self.dtype) for _ in
                                 range(self.num_attention_heads)], dim=0))

        self.params_dict = nn.ParameterDict(param_dict)

    def forward(self, input, mode='weight_input'):
        # TODO: 目前只能支持分解成2个矩阵, 计算等价, 这里最好加一个单元测试以防万一
        # TODO: 前向传播逻辑需要修改
        R0, R1 = self.params_dict['R0'], self.params_dict['R1']
        N0, N1 = self.Ns[0], self.Ns[1]
        input_dtype = input.dtype
        input = input.to(R0.dtype)
        # if mode.endswith("1"):
        #     raise ValueError("Test mode is not support when training")
        if mode == 'weight_input':
            # torch.matmul(W_, Q)
            output = torch.einsum('ij,aik,km->ajm', R0, rearrange(input, 'b (h c)->b h c', h=N0), R1)
            output = rearrange(output, "b h c-> b (h c)")
        elif mode == 'weight_input1':
            # torch.matmul(W_, Q)
            output = torch.matmul(input, torch.kron(R0, R1))
        elif mode == 'weight_output':
            # torch.matmul(Q.T, W)
            output = torch.einsum('i j, i k a, k m -> j m a', R0, rearrange(input, '(b h) c->b h c', b=N0), R1)
            output = rearrange(output, "b h c-> (b h) c")
        elif mode == 'weight_output1':
            # torch.matmul(Q.T, W)
            output = torch.matmul(torch.kron(R0, R1).t(), input)
        elif mode in ['data_input', 'data_embed']:
            # torch.matmul(data, Q)
            assert len(input.shape) == 3
            output = torch.einsum('e j, a e d, d m->a j m', R0, rearrange(input, 'b l (e d)->(b l) e d', e=N0), R1)
            output = output.view_as(input)
        elif mode == 'data_input1':
            # torch.matmul(data, Q)
            assert len(input.shape) == 3
            output = torch.matmul(input, torch.kron(R0, R1))
        elif mode in ['data_qk', ]:
            # [batch_size, head_num, length, embed_dim]
            assert len(input.shape) == 4 and len(R0.shape) == 3 and len(R1.shape) == 3
            output = torch.einsum('h e j, h a e d, h d m->h a j m', R0,
                                  rearrange(input, 'b h l (e d)->h (b l) e d', e=N0), R1)
            output = rearrange(output, "m (b h) t p -> b m h (t p)", b=input.shape[0])
        elif mode in ['data_qk1', ]:
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            output = torch.einsum("b h l d, h d m -> b h l m", input, rotate)
        elif mode == 'weight_v_proj':
            # [num_head, N1xN2, N1xN2]
            assert len(input.shape) == 2 and len(R0.shape) == 3 and len(R1.shape) == 3
            output = torch.einsum('h b j,h b n c,h n m->h j m c', R0,
                                  rearrange(input, '(h b n) c -> h b n c', h=self.num_attention_heads, b=N0), R1)
            output = rearrange(output, 'h j m c -> (h j m) c')
        elif mode == 'weight_v_proj1':
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            # [num_head, N1xN2, N1xN2]
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            # [out_channel, in_channel] -> [num_heads, N1xN2, in_channel]
            output = torch.einsum("b l d, b l c -> b d c", rotate,
                                  rearrange(input, "(b l) c-> b l c", b=self.num_attention_heads))
            output = rearrange(output, "b l c -> (b l) c")
        elif mode == 'weight_o_proj':
            output = torch.einsum('h l j,b h l c,h c m->b h j m', R0,
                                  rearrange(input, 'b (h l c)->b h l c', l=N0, c=N1), R1)
            output = rearrange(output, "b h j m -> b (h j m)")
        elif mode == 'weight_o_proj1':
            # torch.matmul(W_, Q)
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            # [num_head, N1xN2, N1xN2]
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            input = rearrange(input, "b (h c) -> b h c", h=self.num_attention_heads)
            output = torch.einsum('b h c, h c i -> b h i', input, rotate)
            output = rearrange(output, "b h i -> b (h i)")
        else:
            raise NotImplementedError
        output = output.to(input_dtype)
        return output


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


@torch.no_grad()
def init_rotate_to_model(model, dtype=torch.float32, mode='random'):
    Q1 = RotateModule(model.config.hidden_size, dtype=dtype, mode=mode)

    # 给最初的模型应用, 输入的特征, 给embedding的输出用的
    # 在输入lm_head之前需要把特征旋转回来
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers

    # 刚开始的特征需要旋转, 旋转从Embedding开始
    model.model.RotateEmbedding = Q1
    model.lm_head.RotateWeightIn = Q1

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        """这些都是可以offline计算的旋转矩阵, 受到结构的限制, 这里全局都是一个Q"""
        # QKV的权重需要乘一个旋转矩阵进行量化
        layer.self_attn.q_proj.RotateWeightIn = Q1
        layer.self_attn.k_proj.RotateWeightIn = Q1
        layer.self_attn.v_proj.RotateWeightIn = Q1

        # Attention output的输出维度的旋转
        layer.self_attn.o_proj.RotateWeightOut = Q1

        # FFN层的up_proj和gate_proj
        layer.mlp.up_proj.RotateWeightIn = Q1
        layer.mlp.gate_proj.RotateWeightIn = Q1

        # FFN的输出
        layer.mlp.down_proj.RotateWeightOut = Q1

        """剩下的都是需要Online计算的Rotate Matrix"""
        # Q和KVCache种的K可以直接旋转, 两个都旋转Q就好, 因为本身就是会转置的, 这里需要的不同就是, 每个head得对应有自己的旋转矩阵
        # 这个操作是Online的, 这里抵2个Operation
        if model.config.kv_bits < 16:
            Q2 = RotateModule(hidden_size=model.config.hidden_size, dtype=dtype,
                              num_attention_heads=model.config.num_attention_heads, mode=mode)
            layer.self_attn.RotateDataQK = Q2

        # 这里要对应的再去旋转一下down_proj, 因为外部的x已经通过旋转
        Q3 = RotateModule(model.config.intermediate_size, dtype=dtype, mode=mode)
        layer.mlp.down_proj.RotateDataIn = Q3
        layer.mlp.down_proj.RotateWeightIn = Q3

        # TODO: 这里还缺一个Value的旋转和对应的权重的旋转
        Q4 = RotateModule(hidden_size=model.config.hidden_size, dtype=dtype, mode=mode,
                          num_attention_heads=model.config.num_attention_heads)
        layer.self_attn.v_proj.RotateWeightV = Q4
        layer.self_attn.o_proj.RotateWeightO = Q4

    """经过这些操作之后, 在保持教师和学生都是FP32的情况下, 第一步教师和学生的输出应该是相等的"""
    return model


def get_orthogonal_matrix(size, mode, dtype=torch.float32, device='cpu'):
    if mode == 'random':
        return random_orthogonal_matrix(size, dtype, device)
    elif mode == 'hadamard':
        assert is_pow2(size), f"{size} is not power of two"
        coefficient = math.sqrt(2) ** math.log2(size)
        matrix = hadamard(n=size, dtype=np.float32)
        return torch.from_numpy(matrix).to(dtype=dtype, device=device) / coefficient
    else:
        raise ValueError(f'Unknown mode {mode}')

    pass


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def random_orthogonal_matrix(size, dtype=torch.float32, device='cpu'):
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
    random_matrix = torch.randn(size, size, dtype=torch.float32, device=device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(dtype=dtype)


if __name__ == '__main__':
    # TODO: 非常重要!!!!一共有12次旋转
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


    data = torch.randn(2, 128, 256)
    model1 = Attention(dim=256, num_heads=8)
    model2 = FFN(256, 512)
    output = model2(model1(data))

    Q1 = RotateModule(256, num_attention_heads=1)
    Q2 = RotateModule(256, num_attention_heads=8)
    Q3 = RotateModule(256, num_attention_heads=8)
    Q4 = RotateModule(512, num_attention_heads=1)

    # 旋转qkv对应输入的旋转
    data_rotate = Q1(data, mode='data_input')
    model1.q.weight.data = Q1(model1.q.weight.data, mode='weight_input')
    model1.k.weight.data = Q1(model1.k.weight.data, mode='weight_input')
    model1.v.weight.data = Q1(model1.v.weight.data, mode='weight_input')

    # 用在qk^T的旋转
    model1.qk_rotate = Q2

    # 用在v和out_proj上的旋转
    model1.v.weight.data = Q3(model1.v.weight.data, mode='weight_v_proj')
    model1.proj.weight.data = Q3(model1.proj.weight.data, mode='weight_o_proj')

    # 用在out_proj输出的旋转
    model1.proj.weight.data = Q1(model1.proj.weight.data, mode='weight_output')

    # 用在gatemlp输入的 旋转
    model2.gate_proj.weight.data = Q1(model2.gate_proj.weight.data, mode='weight_input')
    model2.up_proj.weight.data = Q1(model2.up_proj.weight.data, mode='weight_input')

    # 用在中间silu之后的online的旋转
    model2.RotateDataIn = Q4
    model2.down_proj.weight.data = Q4(model2.down_proj.weight.data, mode='weight_input')

    # 输出部分的旋转
    model2.down_proj.weight.data = Q1(model2.down_proj.weight.data, mode='weight_output')

    output_test = model2(model1(data_rotate))
    print(f"{torch.abs(Q1(output, mode='data_input') - output_test).abs().max():.10f}")

    # def diff_abs_max(output1, output2):
    #     return (output2 - output1).abs().max()
    #
    #
    # device = 'mps'
    # if device == 'mps':
    #     func = torch.mps.synchronize
    # else:
    #     func = torch.cpu.synchronize
    # num = 10
    # N0_ = 2 ** num
    # N1_ = 1
    # batch_size = 4
    # token_num = 10
    # for i in range(num + 1):
    #     N0 = N0_ // (2 ** i)
    #     N1 = N1_ * (2 ** i)
    #     weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
    #     data = torch.randn(batch_size, token_num, N0 * N1).to(device)
    #     rotate_module = RotateModule(hidden_size=N0 * N1).to(device)
    #
    #     weight_input = rotate_module(weight, mode='weight_input')
    #     weight_input1 = rotate_module(weight, mode='weight_input1')
    #
    #     weight_output = rotate_module(weight, mode='weight_output')
    #     weight_output1 = rotate_module(weight, mode='weight_output1')
    #
    #     data_input = rotate_module(data, mode='data_input')
    #     data_input1 = rotate_module(data, mode='data_input1')
    #
    #     print(f"(N0, N1, diff1, diff2, diff3): "
    #           f"({N0:5}, {N1:5}, "
    #           f"{diff_abs_max(weight_input, weight_input1):.5f}, "
    #           f"{diff_abs_max(weight_output, weight_output1):.5f}, "
    #           f"{diff_abs_max(data_input, data_input1):5f})")
    #
    # for num_attention_heads in [2, 4, 8, 16]:
    #     for i in range(num + 1):
    #         N0 = N0_ // (2 ** i)
    #         N1 = N1_ * (2 ** i)
    #         weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
    #         data = torch.randn(batch_size, num_attention_heads, token_num, (N0 * N1) // num_attention_heads).to(device)
    #         rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)
    #
    #         data_input = rotate_module(data, mode='data_qk')
    #         data_input1 = rotate_module(data, mode='data_qk1')
    #
    #         output1 = torch.einsum('ihtj,ihkj->ihtk', data, data)
    #         output2 = torch.einsum('ihtj,ihkj->ihtk', data_input, data_input)
    #
    #         print(f"(N0, N1, num_attention_heads, diff, attn): "
    #               f"({N0:5}, {N1:5} {num_attention_heads:5}, "
    #               f"{diff_abs_max(data_input, data_input1):5}, "
    #               f"{diff_abs_max(output1, output2):5})")
    #
    # for num_attention_heads in [2, 4, 8, 16]:
    #     for i in range(num + 1):
    #         N0 = N0_ // (2 ** i)
    #         N1 = N1_ * (2 ** i)
    #         weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
    #         rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)
    #         data_input = rotate_module(weight, mode='weight_v')
    #         data_input1 = rotate_module(weight, mode='weight_v1')
    #
    #         output1 = torch.einsum('ij,ih->jh', data_input, data_input)
    #         output2 = torch.einsum('ij,ih->jh', data_input1, data_input1)
    #
    #         print(f"(N0, N1, num_attention_heads, diff, attn): "
    #               f"({N0:5}, {N1:5} {num_attention_heads:5}, "
    #               f"{diff_abs_max(data_input, data_input1):5}, "
    #               f"{diff_abs_max(output1, output2):5})")
    #
    # for num_attention_heads in [2, 4, 8, 16]:
    #     for i in range(num + 1):
    #         N0 = N0_ // (2 ** i)
    #         N1 = N1_ * (2 ** i)
    #         weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
    #         rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)
    #         data_input = rotate_module(weight, mode='weight_o_proj')
    #         data_input1 = rotate_module(weight, mode='weight_o_proj1')
    #
    #         output1 = torch.einsum('ij,ih->jh', data_input, data_input)
    #         output2 = torch.einsum('ij,ih->jh', data_input1, data_input1)
    #
    #         print(output1.shape, output2.shape, data_input.shape, data_input1.shape)
    #         print(f"(N0, N1, num_attention_heads, diff, attn): "
    #               f"({N0:5}, {N1:5} {num_attention_heads:5}, "
    #               f"{diff_abs_max(data_input, data_input1):5}, "
    #               f"{diff_abs_max(output1, output2):5})")
