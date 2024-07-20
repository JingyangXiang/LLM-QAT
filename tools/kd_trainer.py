# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from tools.optimizer import CayleyAdamW, CayleySGD

logger = logging.get_logger(__name__)


class KDModule(PreTrainedModel):
    def __init__(self, student_model, teacher_model):
        super().__init__(student_model.config)

        self.student_model = student_model
        self.teacher_model = teacher_model

    def forward(self, input_ids, labels):
        if self.training:
            student_output = self.student_model(input_ids=input_ids, labels=labels)
            with torch.no_grad():
                teacher_output = self.teacher_model(input_ids=input_ids, labels=labels)
            return student_output, teacher_output
        else:
            student_output = self.student_model(input_ids=input_ids, labels=labels)
            return student_output

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.student_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model, inputs, return_outputs):
        # 损失函数是函数, 状态不会切换, 得根据模型的判断状态
        if model.training:
            student_output, teacher_output = model(**inputs)
            student_output_log_prob = F.log_softmax(student_output.logits, dim=2)
            teacher_output_soft = F.softmax(teacher_output.logits, dim=2)
            loss = self.loss_func(student_output_log_prob, teacher_output_soft)
        else:
            student_output = model(**inputs)
            loss = student_output.loss

        return (loss, student_output) if return_outputs else loss


class KDTrainer(transformers.Trainer):
    """
    主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            loss_func: nn.Module = None,
    ):
        super(KDTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = self.loss_func(model, inputs, return_outputs)
        return loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args)
        optimizer = super().create_optimizer()
        if self.args.optim == 'sgd':
            for param in optimizer.param_groups:
                param['momentum'] = 0.9
                param['weight_decay'] = 0.0001
            print(optimizer)
        return optimizer


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
