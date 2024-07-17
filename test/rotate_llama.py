import copy

import torch
import transformers

from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from tools import rotation_utils, utils
from tools.process_args import process_args

log = utils.get_logger("clm")


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


if __name__ == "__main__":
    model_args, data_args, training_args = process_args()
    device = torch.device("cuda:0")
    log.info("Start to load model...")
    # 旋转过程中使用FP32, 否则误差很大
    # BF16在旋转的那步会直接崩掉, 有问题
    dtype = torch.float16
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = LlamaConfig.from_pretrained(model_args.input_model_filename)
    student_config = copy.deepcopy(config)
    student_config.w_bits = model_args.w_bits
    student_config.a_bits = model_args.a_bits
    student_config.kv_bits = model_args.kv_bits

    message = f"Train with (w_bit, a_bit, kv_bit): ({model_args.w_bits}, {model_args.a_bits}, {model_args.kv_bits})"
    log.info(message)
    student_model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=student_config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    ############################################################################
    # 第一步, 合并RMSNorm
    log.info("Fuse RMSNorm/LayerNorm for student model...")
    rotation_utils.fuse_layer_norms(student_model)
    ############################################################################
    # 第二步, 旋转权重
    log.info("Rotate Embedding and Linear Weight for student model...")
    rotation_utils.init_rotate_to_model(student_model, dtype=dtype)
    ############################################################################
    print(student_model)

    # 这个很正常, 按照正常的训练就可以，不需要任何自定义
    log.info("Start to load tokenizer...")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    log.info("Complete tokenizer loading...")
    seq = "Summer is warm. Winter is cold."
    valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    student_model.cuda()
    student_model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = student_model(valenc).logits
    print(output.mean())
    # 原始网络:     -1.1201
    # 合并Norm之后: -1.1201
    # 旋转后: -1.1328
    # torch.inference_mode

    for name, param in student_model.named_parameters():
        print(name, param.shape)
