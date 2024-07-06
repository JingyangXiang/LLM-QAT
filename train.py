# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Add quantization and knowledge distillialtion
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import math

import torch
import transformers
from torch import distributed as dist
from transformers import AutoModelForCausalLM, default_data_collator, Trainer

from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from utils import datautils, utils
from utils.kd_trainer import KDModule, KDTrainer
from utils.process_args import process_args

log = utils.get_logger("clm")


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    if training_args.qat:
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
            device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
        student_model.eval()
        student_model.cuda()
        log.info("Freeze student model...")
        for param in student_model.parameters():
            param.requires_grad = False
        student_model.config.use_cache = False

    else:
        raise NotImplementedError



    if training_args.use_kd:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model_filename,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
        teacher_model.eval()
        teacher_model.cuda()
        log.info("Freeze teacher model...")
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        model = KDModule(student_model, teacher_model)
    else:
        model = student_model

    log.info("Complete model loading...")

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

    log.info("Start to load datasets...")
    train_dataset, valid_dataset = datautils.get_train_val_dataset(model=model_args.input_model_filename)
    train_data = datautils.CustomJsonDataset(train_dataset)
    valid_data = datautils.CustomJsonDataset(valid_dataset)
    log.info("Complete datasets loading...")
    log.info("Train dataset size: {}, Val dataset size: {}".format(len(train_dataset), len(valid_dataset)))

    if training_args.use_kd:
        from utils.kd_trainer import KDLoss
        trainer = KDTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
            data_collator=default_data_collator,
            loss_func=KDLoss()
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
            data_collator=default_data_collator,
        )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
