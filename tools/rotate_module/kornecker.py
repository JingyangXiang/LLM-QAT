import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn as nn

from tools.rotate_module.utils import get_greatest_common_factor, get_orthogonal_matrix


class KorneckerRotate(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, mode='random', dtype=torch.float32, matrix_cost='min'):
        super(KorneckerRotate, self).__init__()
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
