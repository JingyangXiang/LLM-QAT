# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
input_model_filename=/home/sankuai/dolphinfs_xiangjingyang/huggingface.co/meta-llama/llama-7b-hf
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
train_data_local_path=/home/sankuai/dolphinfs_xiangjingyang/LLM-QAT/gen_data/all_gen.jsonl
torchrun --nproc_per_node=2 --master_port=15001 train.py \
--local_dir "./llama" \
--input_model_filename $input_model_filename \
--output_model_filename "7B-finetuned" \
--train_data_local_path $train_data_local_path \
--eval_data_local_path "wiki2.jsonl" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat True \
--w_bits 8 \
--a_bits 8 \
--kv_bits 8 \
--use_kd True \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
