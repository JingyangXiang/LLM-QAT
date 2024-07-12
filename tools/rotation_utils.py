import gc
import typing

import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import einsum, rearrange, repeat


class RotateModule(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, dtype=torch.float32, matrix_cost='min'):
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
                param_dict[f'R{index}'] = nn.Parameter(get_orthogonal_matrix(N, mode='random', dtype=self.dtype))
            else:
                param_dict[f'R{index}'] = nn.Parameter(
                    repeat(get_orthogonal_matrix(N, mode='random', dtype=self.dtype), 'i j -> k i j',
                           k=self.num_attention_heads))

        self.params_dict = nn.ParameterDict(param_dict)

    def forward(self, input, mode='weight_input'):
        # TODO: 目前只能支持分解成2个矩阵, 计算等价, 这里最好加一个单元测试以防万一
        # TODO: 前向传播逻辑需要修改
        R0, R1 = self.params_dict['R0'], self.params_dict['R1']
        N0, N1 = self.Ns[0], self.Ns[1]
        # for index in range(1, self.len_R):
        #     R0 = torch.kron(R0, self.params_dict[f'R{index}'])
        if mode.endswith("1"):
            raise ValueError("Test mode is not support when training")
        if mode == 'weight_input':
            # torch.matmul(W_, Q)
            output = torch.einsum('ij,aik,km->ajm', R0, rearrange(input, 'b (h c)->b h c', h=N0), R1)
            output = output.reshape(N0 * N1, -1)
        elif mode == 'weight_input1':
            # torch.matmul(W_, Q)
            output = torch.matmul(input, torch.kron(R0, R1))
        elif mode == 'weight_output':
            # torch.matmul(Q.T, W)
            output = torch.einsum('ij,ika,km->jma', R0, rearrange(input, '(h c) b->h c b', h=N0), R1)
            output = output.reshape(-1, N0 * N1)
        elif mode == 'weight_output1':
            # torch.matmul(Q.T, W)
            output = torch.matmul(torch.kron(R0, R1).t(), input)
        elif mode == 'data_input':
            # torch.matmul(data, Q)
            assert len(input.shape) == 3
            output = torch.einsum('ij,aik,km->ajm', R0, rearrange(input, 'b h (t p)->(b h) t p', t=N0), R1)
            output = output.view_as(input)
        elif mode == 'data_input1':
            # torch.matmul(data, Q)
            assert len(input.shape) == 3
            output = torch.matmul(input, torch.kron(R0, R1))
        elif mode in ['data_qk', ]:
            assert len(input.shape) == 4 and len(R0.shape) == 3 and len(R1.shape) == 3
            output = torch.einsum('tij,taik,tkm->tajm', R0, rearrange(input, 'b m h (t p)->m (b h) t p', t=N0), R1)
            output = rearrange(output, "m (b h) t p -> b m h (t p)", b=input.shape[0])
            output = output.view_as(input)
        elif mode in ['data_qk1', ]:
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            output = torch.matmul(input, rotate)
        elif mode == 'weight_v':
            # [num_head, N1xN2, N1xN2]
            assert len(input.shape) == 2 and len(R0.shape) == 3 and len(R1.shape) == 3
            output = torch.einsum('hij,hika,hkm->hjma', R0,
                                  rearrange(input, '(h b n) c -> h b n c', h=self.num_attention_heads, b=N0), R1)
            output = rearrange(output, 'h j m a -> (h j m) a')
            output = output.view_as(input)
        elif mode == 'weight_v1':
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            # [num_head, N1xN2, N1xN2]
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            # [out_channel, in_channel] -> [num_heads, N1xN2, in_channel]
            weight = input.reshape(self.num_attention_heads, -1, input.shape[-1])
            output = torch.matmul(rotate.permute(0, 2, 1), weight)
            output = output.reshape(-1, input.shape[-1])
        elif mode == 'weight_o_proj':
            output = torch.einsum('hij,ahik,hkm->ahjm', R0, rearrange(input, 'b (h o c)->b h o c', o=N0, c=N1), R1)
            output = output.flatten(1)
        elif mode == 'weight_o_proj1':
            # torch.matmul(W_, Q)
            rotate = einsum(R0, R1, "a b c, a d e -> a b d c e")
            # [num_head, N1xN2, N1xN2]
            rotate = rearrange(rotate, 'a b d c e -> a (b d) (c e)')
            input = input.reshape(input.shape[0], self.num_attention_heads, -1)
            output = torch.einsum('ohj,hji->ohi', input, rotate)
            output = output.flatten(1)
        else:
            raise NotImplementedError

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

        # KVCache可以直接旋转, 两个都旋转Q就好, 因为本身就是会转置的, 这里需要的不同就是, 每个head得对应有自己的旋转矩阵
        layer.self_attn.RotateDataQK = RotateModule(hidden_size=model.config.hidden_size,
                                                    num_attention_heads=model.config.num_attention_heads)

        # 这里要对应的再去旋转一下down_proj, 因为外部的x已经通过旋转
        Q_down_proj = RotateModule(model.config.intermediate_size)
        layer.mlp.down_proj.RotateDataIn = Q_down_proj
        layer.mlp.down_proj.RotateWeightIn = Q_down_proj

        # TODO: 这里还缺一个Value的旋转和对应的权重的旋转
        Q_v_o = RotateModule(hidden_size=model.config.hidden_size, num_attention_heads=model.config.num_attention_heads)
        layer.self_attn.RotateDataV = Q_v_o
        layer.self_attn.o_proj = Q_v_o

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


if __name__ == '__main__':
    def diff_abs_max(output1, output2):
        return (output2 - output1).abs().max()


    device = 'mps'
    if device == 'mps':
        func = torch.mps.synchronize
    else:
        func = torch.cpu.synchronize
    num = 10
    N0_ = 2 ** num
    N1_ = 1
    batch_size = 4
    token_num = 10
    for i in range(num + 1):
        N0 = N0_ // (2 ** i)
        N1 = N1_ * (2 ** i)
        weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
        data = torch.randn(batch_size, token_num, N0 * N1).to(device)
        rotate_module = RotateModule(hidden_size=N0 * N1).to(device)

        weight_input = rotate_module(weight, mode='weight_input')
        weight_input1 = rotate_module(weight, mode='weight_input1')

        weight_output = rotate_module(weight, mode='weight_output')
        weight_output1 = rotate_module(weight, mode='weight_output1')

        data_input = rotate_module(data, mode='data_input')
        data_input1 = rotate_module(data, mode='data_input1')

        print(f"(N0, N1, diff1, diff2, diff3): "
              f"({N0:5}, {N1:5}, "
              f"{diff_abs_max(weight_input, weight_input1):.5f}, "
              f"{diff_abs_max(weight_output, weight_output1):.5f}, "
              f"{diff_abs_max(data_input, data_input1):})")

    for num_attention_heads in [2, 4, 8, 16]:
        for i in range(num + 1):
            N0 = N0_ // (2 ** i)
            N1 = N1_ * (2 ** i)
            weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
            data = torch.randn(batch_size, num_attention_heads, token_num, (N0 * N1) // num_attention_heads).to(device)
            rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)

            data_input = rotate_module(data, mode='data_qk')
            data_input1 = rotate_module(data, mode='data_qk1')

            output1 = torch.einsum('ihtj,ihkj->ihtk', data, data)
            output2 = torch.einsum('ihtj,ihkj->ihtk', data_input, data_input)

            print(f"(N0, N1, num_attention_heads, diff, attn): "
                  f"({N0:5}, {N1:5} {num_attention_heads:5}, "
                  f"{diff_abs_max(data_input, data_input1):5}, "
                  f"{diff_abs_max(output1, output2):5})")

    for num_attention_heads in [2, 4, 8, 16]:
        for i in range(num + 1):
            N0 = N0_ // (2 ** i)
            N1 = N1_ * (2 ** i)
            weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
            rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)
            data_input = rotate_module(weight, mode='weight_v')
            data_input1 = rotate_module(weight, mode='weight_v1')

            output1 = torch.einsum('ij,ih->jh', data_input, data_input)
            output2 = torch.einsum('ij,ih->jh', data_input1, data_input1)

            print(f"(N0, N1, num_attention_heads, diff, attn): "
                  f"({N0:5}, {N1:5} {num_attention_heads:5}, "
                  f"{diff_abs_max(data_input, data_input1):5}, "
                  f"{diff_abs_max(output1, output2):5})")

    for num_attention_heads in [2, 4, 8, 16]:
        for i in range(num + 1):
            N0 = N0_ // (2 ** i)
            N1 = N1_ * (2 ** i)
            weight = torch.randn(N0 * N1, N0 * N1, device=device, dtype=torch.float32)
            rotate_module = RotateModule(hidden_size=N0 * N1, num_attention_heads=num_attention_heads).to(device)
            data_input = rotate_module(weight, mode='weight_o_proj')
            data_input1 = rotate_module(weight, mode='weight_o_proj1')

            output1 = torch.einsum('ij,ih->jh', data_input, data_input)
            output2 = torch.einsum('ij,ih->jh', data_input1, data_input1)

            print(output1.shape, output2.shape, data_input.shape, data_input1.shape)
            print(f"(N0, N1, num_attention_heads, diff, attn): "
                  f"({N0:5}, {N1:5} {num_attention_heads:5}, "
                  f"{diff_abs_max(data_input, data_input1):5}, "
                  f"{diff_abs_max(output1, output2):5})")
