import gc

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers) -> None:
    """fuse the linear operations in Layernorm into the adjacent linear blocks."""
    # 将RMSNorm/LayerNorm的权重fuse到下一层的输入通道上
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype).clone()

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            raise NotImplementedError

    layernorm.weight.data = torch.ones_like(layernorm.weight)


@torch.no_grad()
def fuse_layer_norms(model):
    layers = model.model1.layers

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # 每个
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        # 每个LlamaDecoderLayer的AttentionLayer
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    fuse_ln_linear(model.model1.norm, [model.lm_head])


def random_orthogonal_matrix(size, device, dtype):
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
    random_matrix = torch.randn(size, size, dtype=torch.float32)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(device, dtype=dtype)


@torch.no_grad()
def get_orthogonal_matrix(size, mode, dtype, device='cpu'):
    if mode == 'random':
        return random_orthogonal_matrix(size, device, dtype)
    else:
        raise ValueError(f'Unknown mode {mode}')


@torch.no_grad()
def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    W = model.model1.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


@torch.no_grad()
def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


@torch.no_grad()
def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float32)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


@torch.no_grad()
def rotate_mlp_output(layer, Q):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        raise NotImplementedError


@torch.no_grad()
def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float32)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


@torch.no_grad()
def rotate_attention_output(layer, Q) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float32)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


@torch.no_grad()
def rotate_model(model, dtype):
    Q = get_orthogonal_matrix(model.config.hidden_size, 'random', dtype, model.device)
    # Q = torch.eye(model.config.hidden_size, device=model.device, dtype=torch.float32)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model1.layers
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q)
        rotate_attention_output(layers[idx], Q)
        rotate_mlp_input(layers[idx], Q)
        rotate_mlp_output(layers[idx], Q)
        gc.collect()
        torch.cuda.empty_cache()


pretrained_model_name_or_path = ('/home/xjy/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/'
                                 'snapshots/01c7f73d771dfac7d292323805ebc428287df4f9')
teacher_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map='cuda:0',
)
print(teacher_model.device)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
)
teacher_model.config.use_cache = False

seq = "Summer is warm. Winter is cold."
valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to("cuda:0")
with torch.no_grad():
    output_before = teacher_model(valenc).get("logits", None)
print(output_before.flatten()[:10])

fuse_layer_norms(teacher_model)
rotate_model(teacher_model, dtype=torch.float32)

with torch.no_grad():
    output_after = teacher_model(valenc).get("logits", None)
print(output_after.flatten()[:10])

print((output_after - output_before).abs().max())

#