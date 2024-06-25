# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    "/home/sankuai/dolphinfs_xiangjingyang/huggingface.co/meta-llama/llama-7b-hf")
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    "/home/sankuai/dolphinfs_xiangjingyang/huggingface.co/meta-llama/llama-7b-hf")
model = model.cuda()
print("Model loaded!")

n_vocab = 500  # number of initial tokens for synthesizing data on each GPU.

i_start = sys.argv[1]
if os.path.exists("gen_data/gen.chunk." + str(i_start).zfill(2) + ".jsonl"):
    with open("gen_data/gen.chunk." + str(i_start).zfill(2) + ".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for j in range(3 + outer_loop, 6):
    # outer_loop: 0 -> j: 3, 4, 5
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start) + 1) * n_vocab):
        print(i)
        # 这个数据的生成感觉也很随便啊......
        input_ids = torch.tensor([[i]]).cuda()
        print("generating")
        # 假设生成的顺序
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
        # 多次生成数据
        outputs = model.generate(outputs1, do_sample=True, max_length=2048)
        # 获取生成的文本
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text": gen_text[0]}
        with open("gen_data/gen.chunk." + str(i_start).zfill(2) + ".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
