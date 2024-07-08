import torch


class WeightPerChannelFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits):
        # w: (out_features, in_features)
        scales = weight.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        weight = torch.round(weight / scales) * scales
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class WeightPerTensorFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, num_bits):
        scales = weight.abs().max()
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        weight = torch.round(weight / scales) * scales
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActPerTokenFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, num_bits):
        t_shape = activation.shape
        activation = activation.view(-1, t_shape[-1])
        scales = activation.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        activation = torch.round(activation / scales) * scales

        return activation

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActPerTensorFakeQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, num_bits):
        t_shape = activation.shape
        activation = activation.view(-1, t_shape[-1])
        scales = activation.abs().max()
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        activation = torch.round(activation / scales) * scales

        return activation.reshape(t_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
