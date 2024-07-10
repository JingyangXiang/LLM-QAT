# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F

from tools.fake_quant import ActPerTensorFakeQuantizer, ActPerTokenFakeQuantizer, WeightPerChannelFakeQuantizer, \
    WeightPerTensorFakeQuantizer


class QuantizeLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias=False,
            symmetric=True,
            w_bits=32,
            a_bits=32,
            weight_quant='per_channel',
            act_quant="per_token"
    ):
        super(QuantizeLinear, self).__init__(in_features, out_features, bias=bias)

        self.symmetric = symmetric
        assert self.symmetric, "only symmetric is supported"
        self.w_bits = w_bits
        self.a_bits = a_bits
        if weight_quant == 'per_channel':
            self.weight_quant = WeightPerChannelFakeQuantizer.apply
        elif weight_quant == 'per_tensor':
            self.weight_quant = WeightPerTensorFakeQuantizer.apply
        else:
            raise NotImplementedError

        if act_quant == 'per_token':
            self.act_quant = ActPerTokenFakeQuantizer.apply
        elif act_quant == 'per_tensor':
            self.act_quant = ActPerTensorFakeQuantizer.apply
        else:
            raise NotImplementedError
        assert bias is False, "only bias=False is supported"

    def forward(self, input):

        # quantize weight
        assert len(self.weight.size()) == 2
        weight = self.weight

        if hasattr(self, "RotateWeightIn"):
            weight = weight @ self.RotateWeightIn

        if hasattr(self, "RotateWeightOut"):
            weight = self.RotateWeightOut.t() @ weight

        if self.w_bits not in [16, 32]:
            weight = self.weight_quant(weight, self.w_bits)

        # Quantize inputs
        if self.a_bits not in [16, 32]:
            input = self.act_quant(input, self.a_bits)

        return F.linear(input, weight, self.bias)
