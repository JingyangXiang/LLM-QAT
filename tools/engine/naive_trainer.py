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
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from tools.engine.optimizer import create_custom_optimzer

logger = logging.get_logger(__name__)


class NaiveTrainer(transformers.Trainer):
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
    ):
        super(NaiveTrainer, self).__init__(
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
        self.loss_func = LabelSmoother(epsilon=0.)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
            return loss
        else:
            outputs = model(**inputs)
            loss = self.loss_func(outputs, inputs.get("labels"), shift_labels=True)
            return (loss, outputs) if return_outputs else loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args)
        optimizer = super().create_optimizer()
        return optimizer
