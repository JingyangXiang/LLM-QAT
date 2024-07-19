from utils.quant_funcs import *


class WeightPerChannelFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits):
        # w: (out_features, in_features)
        weight = quantize_weight_per_channel_absmax(weight, num_bits)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class WeightPerTensorFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits):
        weight = quantize_weight_per_tensor_absmax(weight, num_bits)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActPerTokenFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, num_bits):
        activation = quantize_activation_per_token_absmax(activation, num_bits)
        return activation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActPerTensorFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, num_bits):
        activation = quantize_activation_per_tensor_absmax(activation, num_bits)
        return activation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActPerGroupFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, num_bits, group_size):
        activation = quantize_activation_per_group_absmax(activation, num_bits, group_size)
        return activation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
