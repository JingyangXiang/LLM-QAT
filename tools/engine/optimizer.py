from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments

from tools.optimizer import CayleyAdamW, CayleySGD


def create_custom_optimzer(
        model: "PreTrainedModel",
        training_args: "TrainingArguments"
) -> Optional["torch.optim.Optimizer"]:
    # 推荐使cayley_SGD
    learning_rate = training_args.learning_rate
    weight_decay = training_args.weight_decay
    rotate_paramaters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"Trainable: {name}, shape: {param.shape}")
            rotate_paramaters.append(param)
            assert any([f"R{i}" in name for i in range(10)])
            assert len(param.shape) in [1, 2, 3]

    if training_args.optim == 'cayley_adamw':
        beta1 = training_args.adam_beta1
        beta2 = training_args.adam_beta2
        return CayleyAdamW(params=rotate_paramaters, betas=(beta1, beta2), lr=learning_rate, weight_decay=weight_decay)

    elif training_args.optim == 'cayley_sgd':
        return CayleySGD(params=rotate_paramaters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        pass
