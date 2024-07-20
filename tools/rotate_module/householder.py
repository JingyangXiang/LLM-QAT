import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HouseholderModule(nn.Module):
    # 给全局的旋转矩阵
    def __init__(self, hidden_size, num_attention_heads=1, mode='random', dtype=torch.float32, matrix_cost='min'):
        super(HouseholderModule, self).__init__()
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
        self.register_buffer("eye", torch.eye(self.hidden_size // self.num_attention_heads, dtype=self.dtype))

    def register_rotate_matrix(self):
        param_dict = {}
        if self.num_attention_heads == 1:
            param_dict["R0"] = nn.Parameter(torch.randn(self.hidden_size // self.num_attention_heads, dtype=self.dtype))
        else:
            param_dict["R0"] = nn.Parameter(
                torch.randn(self.num_attention_heads, self.hidden_size // self.num_attention_heads, dtype=self.dtype))
        self.params_dict = nn.ParameterDict(param_dict)

    def forward(self, input, mode='weight_input'):
        # TODO: 目前只能支持分解成2个矩阵, 计算等价, 这里最好加一个单元测试以防万一
        # TODO: 前向传播逻辑需要修改
        unit_vector = F.normalize(self.params_dict['R0'], dim=-1)
        if len(unit_vector.shape) == 1:
            R0 = self.eye - 2 * torch.einsum('i,j->ij', unit_vector, unit_vector)
        else:
            R0 = self.eye - 2 * torch.einsum('ki,kj->kij', unit_vector, unit_vector)

        input_dtype = input.dtype
        input = input.to(R0.dtype)

        if mode == 'weight_input':
            # torch.matmul(W_, Q)
            assert len(unit_vector.shape) == 1
            output = input - 2 * torch.einsum("jm,m,n->jn", input, unit_vector, unit_vector)
        elif mode == 'weight_output':
            # torch.matmul(Q.T, W)
            assert len(unit_vector.shape) == 1
            output = input - 2 * torch.einsum('i,j,jm->im', unit_vector, unit_vector, input)
        elif mode in ['data_input', "data_embed"]:
            # torch.matmul(data, Q)
            assert len(input.shape) == 3 and len(unit_vector.shape) == 1
            output = input - 2 * torch.einsum('bhj,j,i->bhi', input, unit_vector, unit_vector)
        elif mode in ['data_qk', ]:
            assert len(unit_vector.shape) == 2
            output = input - 2 * torch.einsum('b h l d, h d, h j -> b h l j', input, unit_vector, unit_vector)
        elif mode == 'weight_v_proj':
            # [num_head, N1xN2, N1xN2]
            # [out_channel, in_channel] -> [num_heads, N1xN2, in_channel]
            output = rearrange(input, "(b l) c-> b l c", b=self.num_attention_heads)
            output = output - 2 * torch.einsum('b l, b d, b l c -> b d c', unit_vector, unit_vector, output)
            output = rearrange(output, "b l c -> (b l) c")
        elif mode == 'weight_o_proj':
            # torch.matmul(W_, Q)
            # [num_head, N1xN2, N1xN2]
            output = rearrange(input, "b (h c) -> b h c", h=self.num_attention_heads)
            output = output - 2 * torch.einsum('b h c, h c, h i -> b h i', output, unit_vector, unit_vector)
            output = rearrange(output, "b h i -> b (h i)")
        else:
            raise NotImplementedError("Unsupported mode {}".format(mode))
        output = output.to(input_dtype)
        return output


if __name__ == "__main__":
    shape = 32
    data = torch.randn(shape)
    weight = torch.randn(shape * 4, shape)
    unit_vector = F.normalize(data, dim=-1)
    eye = torch.eye(shape)
    rotate_matrix = eye - 2 * torch.einsum('i,j->ij', unit_vector, unit_vector)

    weight_output1 = weight @ rotate_matrix
    weight_output2 = 2 * torch.einsum("jm,m,n->jn", weight, unit_vector, unit_vector)
    weight_output2 = weight - weight_output2
    print((weight_output1 - weight_output2).abs().max())
