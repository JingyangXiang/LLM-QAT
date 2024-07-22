from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel

from tools.optimizer import CayleyAdamW, CayleySGD


def create_custom_optimzer(
        model: "PreTrainedModel",
        training_args,
) -> Optional["torch.optim.Optimizer"]:
    learning_rate = training_args.learning_rate
    weight_decay = training_args.weight_decay
    momentum = training_args.momentum
    paramteters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert any([f"R{i}" in name for i in range(10)])
            assert len(param.shape) in [1, 2, 3]
            paramteters.append(param)
    if training_args.optim == 'cayley_adamw':
        beta1 = training_args.adam_beta1
        beta2 = training_args.adam_beta2
        return CayleyAdamW(params=paramteters, betas=(beta1, beta2), lr=learning_rate, weight_decay=weight_decay)
    elif training_args.optim == 'cayley_sgd':
        return CayleySGD(params=paramteters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif training_args.optim == 'sgd':
        return torch.optim.SGD(paramteters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"{training_args.optim} is not support...")
