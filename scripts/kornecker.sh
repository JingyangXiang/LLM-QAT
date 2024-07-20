# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export input_model_filename=/home/sankuai/dolphinfs_xiangjingyang/huggingface.co/meta-llama/Llama-2-7b-hf
python3 train.py \
--local_dir "./result/llama-7b-hf/" \
--input_model_filename $input_model_filename \
--output_model_filename "llama-7b-hf" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 False \
--log_on_each_node False \
--logging_dir ./result/llama-7b-hf \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--gradient_checkpointing True \
--qat True \
--w_bits $1 \
--a_bits $2 \
--kv_bits $3 \
--use_kd True \
--cayley_optim cayley_sgd \
--module_type kornecker \
--save_safetensors False