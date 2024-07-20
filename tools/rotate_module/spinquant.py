import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from tools.rotate_module.utils import get_orthogonal_matrix


class SpinRotateModule(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, mode='random', dtype=torch.float32, matrix_cost='min'):
        super(SpinRotateModule, self).__init__()
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
        self.register_rotate_matrix()

    def register_rotate_matrix(self):
        self.Ns = [self.hidden_size // self.num_attention_heads, ]
        self.len_R = len(self.Ns)
        assert np.prod(self.Ns) == self.hidden_size // self.num_attention_heads

        param_dict = {}

        for index, N in enumerate(self.Ns):
            if self.num_attention_heads == 1:
                param_dict[f'R{index}'] = nn.Parameter(get_orthogonal_matrix(N, mode=self.mode, dtype=self.dtype))
            else:
                param_dict[f'R{index}'] = nn.Parameter(
                    torch.stack([get_orthogonal_matrix(N, mode=self.mode, dtype=self.dtype) for _ in
                                 range(self.num_attention_heads)], dim=0))

        self.params_dict = nn.ParameterDict(param_dict)

    def forward(self, input, mode='weight_input'):
        # TODO: 目前只能支持分解成2个矩阵, 计算等价, 这里最好加一个单元测试以防万一
        # TODO: 前向传播逻辑需要修改
        R0 = self.params_dict['R0']
        input_dtype = input.dtype
        input = input.to(R0.dtype)

        if mode == 'weight_input':
            # torch.matmul(W_, Q)
            output = torch.matmul(input, R0)
        elif mode == 'weight_output':
            # torch.matmul(Q.T, W)
            output = torch.matmul(R0.t(), input)
        elif mode in ['data_input', "data_embed"]:
            # torch.matmul(data, Q)
            assert len(input.shape) == 3
            output = torch.matmul(input, R0)
        elif mode in ['data_qk', ]:
            output = torch.einsum("b h l d, h d m -> b h l m", input, R0)
        elif mode == 'weight_v_proj':
            # [num_head, N1xN2, N1xN2]
            # [out_channel, in_channel] -> [num_heads, N1xN2, in_channel]
            output = torch.einsum("b l d, b l c -> b d c", R0,
                                  rearrange(input, "(b l) c-> b l c", b=self.num_attention_heads))
            output = rearrange(output, "b l c -> (b l) c")
        elif mode == 'weight_o_proj':
            # torch.matmul(W_, Q)
            # [num_head, N1xN2, N1xN2]
            input = rearrange(input, "b (h c) -> b h c", h=self.num_attention_heads)
            output = torch.einsum('b h c, h c i -> b h i', input, R0)
            output = rearrange(output, "b h i -> b (h i)")
        else:
            raise NotImplementedError("Unsupported mode {}".format(mode))
        output = output.to(input_dtype)
        return output
