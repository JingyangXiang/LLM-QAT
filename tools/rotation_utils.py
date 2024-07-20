import gc
import typing

import torch
import tqdm

from tools.rotate_module.householder import HouseholderModule
from tools.rotate_module.kornecker import KorneckerRotate
from tools.rotate_module.misc import Attention, FFN
from tools.rotate_module.spinquant import SpinRotateModule


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


@torch.no_grad()
def init_rotate_to_model(model, dtype=torch.float32, mode='random', module_type='kornecker'):
    #
    rotate_modules = {'kornecker': KorneckerRotate, "spin": SpinRotateModule, "householder": HouseholderModule}
    module = rotate_modules[module_type]
    Q1 = module(model.config.hidden_size, dtype=dtype, mode=mode)

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
            Q2 = module(hidden_size=model.config.hidden_size, dtype=dtype,
                        num_attention_heads=model.config.num_attention_heads, mode=mode)
            layer.self_attn.RotateDataQK = Q2

        # 这里要对应的再去旋转一下down_proj, 因为外部的x已经通过旋转
        Q3 = module(model.config.intermediate_size, dtype=dtype, mode=mode)
        layer.mlp.down_proj.RotateDataIn = Q3
        layer.mlp.down_proj.RotateWeightIn = Q3

        # TODO: 这里还缺一个Value的旋转和对应的权重的旋转
        Q4 = module(hidden_size=model.config.hidden_size, dtype=dtype, mode=mode,
                    num_attention_heads=model.config.num_attention_heads)
        layer.self_attn.v_proj.RotateWeightV = Q4
        layer.self_attn.o_proj.RotateWeightO = Q4

    """经过这些操作之后, 在保持教师和学生都是FP32的情况下, 第一步教师和学生的输出应该是相等的"""
    return model


if __name__ == '__main__':
    # TODO: 非常重要!!!!一共有12次旋转
    rotate_modules = [KorneckerRotate, SpinRotateModule, HouseholderModule]
    for rotate_module in rotate_modules:
        print(f"==> {rotate_module.__name__}")
        data = torch.randn(2, 128, 256)
        model1 = Attention(dim=256, num_heads=8)
        model2 = FFN(256, 512)
        output = model2(model1(data))

        Q1 = rotate_module(256, num_attention_heads=1)
        Q2 = rotate_module(256, num_attention_heads=8)
        Q3 = rotate_module(256, num_attention_heads=8)
        Q4 = rotate_module(512, num_attention_heads=1)

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
