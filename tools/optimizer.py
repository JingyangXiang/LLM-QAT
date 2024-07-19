""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def qr_retraction3d(tan_vec):  # tan_vec, p-by-n, p <= n
    # [p, n] = tan_vec.size()
    tan_vec = tan_vec.permute(0, 2, 1)
    q, r = torch.linalg.qr(tan_vec)

    d = torch.diagonal(r, dim1=-2, dim2=-1)
    ph = d.sign()
    q = torch.einsum("ij,ikj->ikj", ph, q)
    q = q.permute(0, 2, 1)
    return q


def qr_retraction2d(tan_vec):  # tan_vec, p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()

    return q


def CayleyLoop(X, W, tan_vec, t):
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.einsum("ijk, ikm->ijm", W, 0.5 * (X + Y))
    return Y.permute(0, 2, 1)


def matrix_norm_one(W):
    # 感觉还是为了稳定性, 求单个维度的绝对值和, 然后得到最大值
    out = torch.abs(W)
    out = torch.sum(out, dim=1, keepdim=True)
    # 暂时可以放着不动
    # out = torch.max(out.flatten(1), dim=-1).values
    out = torch.max(out)
    return out


class CayleySGD(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr, momentum=0.9, dampening=0, eps=1e-8, weight_decay=0,
                 nesterov=False, stiefel=True, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip, eps=eps)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CayleySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CayleySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 理论上来说, p一直都应当是一个正交矩阵, p_normalize和p不应该有明显的变化
                p_normalize = F.normalize(p, dim=-1, p=2)

                # 这里扩展成三维
                if len(p_normalize.shape) == 2:
                    p_normalize = p_normalize.unsqueeze(0)

                # 严格一些是, p必须方阵
                assert p_normalize.size()[1] <= p_normalize.size()[2] and len(p_normalize.shape) == 3, \
                    f'p_normalize.size()[1]> p_normalize.size()[2] is not supported'

                if stiefel and p_normalize.size()[1] <= p_normalize.size()[2]:

                    weight_decay = group['weight_decay']
                    assert weight_decay == 0

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        p_normalize = qr_retraction3d(p_normalize)
                    g = p.grad.data.view_as(p_normalize)

                    lr = group['lr']

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(g)

                    V = param_state['momentum_buffer']
                    V = momentum * V - g
                    MX = torch.einsum("mij, mik -> mjk", V, p_normalize)
                    XMX = torch.einsum('mij, mjk -> mik', p_normalize, MX)
                    XXMX = torch.einsum('mij, mik -> mjk', p_normalize, XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.permute(0, 2, 1)
                    t = 0.5 * 2 / (matrix_norm_one(W) + group['eps'])
                    alpha = min(t, lr)

                    p_new = CayleyLoop(p_normalize.permute(0, 2, 1), W, V.permute(0, 2, 1), alpha)
                    V_new = torch.einsum("mij, mkj->mki", W, p_normalize)  # n-by-p

                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)
                else:
                    raise NotImplementedError

        return loss


class CayleyAdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0., amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(CayleyAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CayleyAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 理论上来说, p一直都应当是一个正交矩阵, p_normalize和p不应该有明显的变化
                p_normalize = F.normalize(p, dim=-1, p=2)

                # 这里扩展成三维
                if len(p_normalize.shape) == 2:
                    p_normalize = p_normalize.unsqueeze(0)

                # 严格一些是, p必须方阵
                assert p_normalize.size()[1] <= p_normalize.size()[2] and len(p_normalize.shape) == 3, \
                    f'p_normalize.size()[0]> p_normalize.size()[1] is not supported'

                if random.randint(1, 101):
                    # 就进行QR分解, 这个时候p_normalize肯定是正交的了
                    # 虽然感觉没有必要, 因为一开始约束的时候肯定就是正交初始化, 这个算法一直是正交的
                    p_normalize = qr_retraction3d(p_normalize)

                # Perform step weight decay
                # 在CayleyAdamW中不支持weight_decay, 因为无法处理这个数据
                assert group['weight_decay'] == 0, 'CayleyAdamW does not support weight decay'
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # 不支持amsgrad, 目前不知道这个机制是干嘛的, 直接关掉
                amsgrad = group['amsgrad']
                assert amsgrad is False, 'CayleyAdamW does not support amsgrad'

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # 注意这里没有用梯度的转置, 源码里面用了
                    # https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py
                    state['exp_avg'] = torch.zeros_like(p_normalize)
                    # Exponential moving average grad norm
                    state['exp_avg_norm'] = torch.zeros(p_normalize.shape[0])
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_norm'] = torch.zeros(p_normalize.shape[0])

                # 获取存储的一阶矩和二阶矩
                exp_avg, exp_avg_norm = state['exp_avg'], state['exp_avg_norm']
                if amsgrad:
                    max_exp_avg_norm = state['max_exp_avg_norm']
                beta1, beta2 = group['betas']

                # 记录step, 这个应该没啥用
                state['step'] += 1
                # bias_correction1 和 bias_correction2
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                # 先计算指数移动平均(EMA)出来的相关参数, 需要注意的是这里是原地更新
                grad = grad.view_as(p_normalize)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_norm.mul_(beta2).addcmul_(torch.norm(grad, dim=[1, 2]), torch.norm(grad, dim=[1, 2]),
                                                  value=1 - beta2)

                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_norm, exp_avg_norm, out=max_exp_avg_norm)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = (max_exp_avg_norm.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # else:
                #     denom = (exp_avg_norm.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_norm_hat = exp_avg_norm / bias_correction2

                # X^T X G X
                # MX = torch.matmul(exp_avg_hat.t(), p_normalize)
                # XMX = torch.matmul(p_normalize, MX)
                # XXMX = torch.matmul(p_normalize.t(), XMX)

                MX = torch.einsum("hij,hik->hjk", exp_avg_hat, p_normalize)
                XXMX = torch.einsum('lij, lik, lhk, lhm->ljm', p_normalize, p_normalize, exp_avg_hat, p_normalize)

                W_hat = MX - 0.5 * XXMX
                W = (W_hat - W_hat.permute(0, 2, 1)) / exp_avg_norm_hat.add(group['eps']).sqrt().reshape(-1, 1, 1)

                t = 0.5 * 2 / (matrix_norm_one(W) + group['eps'])
                # print(matrix_norm_one(W), group['lr'])
                alpha = min(t, group['lr'])

                # 这里是进行根据CayleyLoop得到的正交的新参数fixed-point
                p_new = CayleyLoop(p_normalize.permute(0, 2, 1), W, exp_avg.permute(0, 2, 1), -alpha)
                p.data.copy_(p_new.view(p.size()))

                exp_avg_new = torch.einsum('ijk,ikm->ijm', p_normalize, W) * \
                              exp_avg_norm_hat.add(group['eps']).sqrt().reshape(-1, 1, 1) * (1 - beta1 ** state['step'])
                exp_avg.copy_(exp_avg_new)
        return loss


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        assert bias == False
        tensor = torch.randn(in_channel, int(math.sqrt(out_channel)), int(math.sqrt(out_channel)))
        self.weight = nn.Parameter(tensor)
        for index in range(self.weight.shape[0]):
            nn.init.orthogonal_(self.weight[index])

    def forward(self, input):
        return input @ self.weight.flatten(1)


if __name__ == '__main__':
    # 优化器测试，暂时没发现致命的问题
    batch_size = 4
    in_channel = 32
    output_channel = 64
    model1 = Linear(in_channel, output_channel, bias=False)
    with torch.no_grad():
        print(torch.diagonal(model1.weight @ model1.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum())
    optim1 = CayleyAdamW(model1.parameters(), lr=0.1)
    data1 = torch.randn(batch_size, in_channel)
    for i in range(10):
        optim1.zero_grad()
        output = (model1(data1).square()).mean()
        output.backward()
        optim1.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diagonal(model1.weight @ model1.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum().item():.2f}, "
                  f"weight: {torch.mean(model1.weight.flatten()[:6]):.5f}")

    del model1

    model2 = Linear(in_channel, output_channel, bias=False)
    with torch.no_grad():
        print(torch.diagonal(model2.weight @ model2.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum())
    data2 = torch.randn(batch_size, in_channel)
    optim2 = CayleySGD(model2.parameters(), lr=0.1, momentum=0.9)
    for i in range(10):
        optim2.zero_grad()
        output = (model2(data2).abs()).mean()
        output.backward()
        optim2.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diagonal(model2.weight @ model2.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum().item():.2f}, "
                  f"weight: {torch.mean(model2.weight.flatten()[:6]):.5f}")

    model3 = Linear(in_channel, output_channel, bias=False)
    with torch.no_grad():
        print(torch.diagonal(model3.weight @ model3.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum())
    data3 = torch.randn(batch_size, in_channel)
    optim3 = torch.optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)
    for i in range(10):
        optim3.zero_grad()
        output = (model3(data3).abs()).mean()
        output.backward()
        optim3.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diagonal(model3.weight @ model3.weight.permute(0, 2, 1), dim1=-2, dim2=-1).sum().item():.2f}, "
                  f"weight: {torch.mean(model3.weight.flatten()[:6]):.5f}")
