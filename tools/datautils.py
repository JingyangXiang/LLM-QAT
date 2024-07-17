# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

logger = logging.get_logger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(model, calib_dataset='wikitext', nsamples=800, seqlen=2048, seed=0):
    assert calib_dataset == 'wikitext'
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    train_enc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    test_enc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    logger.info("get_wikitext2 for training")
    train_loader = []
    for _ in range(nsamples):
        i = random.randint(0, train_enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp.squeeze(), tar.squeeze()))

    logger.info("get_wikitext2 for testing")
    test_loader = []
    for i in range(0, test_enc.input_ids.numel(), seqlen):
        j = i + seqlen
        inp = test_enc.input_ids[:, i:j]
        tar = inp.clone()
        test_loader.append((inp.squeeze(), tar.squeeze()))

    return train_loader, test_loader


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_iter = [
            dict(input_ids=self.dataset[i][0], labels=self.dataset[i][0])
            for i in range(len(self.dataset))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return dict(input_ids=self.dataset[i][0], labels=self.dataset[i][1])

    def __iter__(self):
        return iter(self.data_iter)
