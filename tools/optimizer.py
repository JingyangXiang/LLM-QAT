""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def CayleyLoop(X, W, tan_vec, t):
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


def matrix_norm_one(W):
    # 感觉还是为了稳定性, 求单个维度的绝对值和, 然后得到最大值
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
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

                # 严格一些是, p必须方阵
                assert p_normalize.size()[0] <= p_normalize.size()[1] and len(p_normalize.shape) == 2, \
                    f'p_normalize.size()[0]> p_normalize.size()[1] is not supported'

                if stiefel and p_normalize.size()[0] <= p_normalize.size()[1]:

                    weight_decay = group['weight_decay']
                    assert weight_decay == 0

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        p_normalize, r_temp = torch.linalg.qr(p_normalize)
                        p_normalize *= torch.sign(torch.diag(r_temp)).unsqueeze(0)

                    g = p.grad.data.view(p.size()[0], -1)

                    lr = group['lr']

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(g)

                    V = param_state['momentum_buffer']
                    V = momentum * V - g
                    MX = torch.einsum("ij, ik -> jk", V, p_normalize)
                    XMX = torch.einsum('ij, jk -> ik', p_normalize, MX)
                    XXMX = torch.einsum('ij, ik -> jk', p_normalize, XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + group['eps'])
                    alpha = min(t, lr)

                    p_new = CayleyLoop(p_normalize.t(), W, V.t(), alpha)
                    V_new = torch.einsum("ij, kj->ki", W, p_normalize)  # n-by-p

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

                # 严格一些是, p必须方阵
                assert p_normalize.size()[0] <= p_normalize.size()[1] and len(p_normalize.shape) == 2, \
                    f'p_normalize.size()[0]> p_normalize.size()[1] is not supported'

                if random.randint(1, 101) == 1:
                    # 就进行QR分解, 这个时候p_normalize肯定是正交的了
                    # 虽然感觉没有必要, 因为一开始约束的时候肯定就是正交初始化, 这个算法一直是正交的
                    p_normalize, r_temp = torch.linalg.qr(p_normalize)
                    p_normalize *= torch.sign(torch.diag(r_temp)).unsqueeze(0)

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
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average grad norm
                    state['exp_avg_norm'] = torch.zeros([1])
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_norm'] = torch.zeros([1])

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
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_norm.mul_(beta2).addcmul_(torch.norm(grad), torch.norm(grad), value=1 - beta2)

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

                MX = torch.einsum("ij,ik->jk", exp_avg_hat, p_normalize)
                XXMX = torch.einsum('ij, ik, hk, hm->jm', p_normalize, p_normalize, exp_avg_hat, p_normalize)

                W_hat = MX - 0.5 * XXMX
                W = (W_hat - W_hat.t()) / exp_avg_norm_hat.add(group['eps']).sqrt()

                t = 0.5 * 2 / (matrix_norm_one(W) + group['eps'])
                # print(matrix_norm_one(W), group['lr'])
                alpha = min(t, group['lr'])

                # 这里是进行根据CayleyLoop得到的正交的新参数fixed-point
                p_new = CayleyLoop(p_normalize.t(), W, exp_avg.t(), -alpha)
                p.data.copy_(p_new.view(p.size()))

                exp_avg_new = torch.matmul(p_normalize, W) * \
                              exp_avg_norm_hat.add(group['eps']).sqrt() * (1 - beta1 ** state['step'])
                exp_avg.copy_(exp_avg_new)
        return loss


if __name__ == '__main__':
    # 优化器测试，暂时没发现致命的问题
    batch_size = 4
    in_channel = 32
    output_channel = in_channel // 2
    model1 = nn.Linear(in_channel, output_channel, bias=False)
    torch.nn.init.orthogonal_(model1.weight)
    with torch.no_grad():
        print(torch.diag(model1.weight @ model1.weight.t()))
    optim1 = CayleyAdamW(model1.parameters(), lr=0.1)
    data1 = torch.randn(batch_size, in_channel)
    for i in range(10):
        optim1.zero_grad()
        output = (model1(data1).square()).mean()
        output.backward()
        optim1.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diag(model1.weight @ model1.weight.t()).sum().item():.2f}, "
                  f"weight: {torch.mean(model1.weight.flatten()[:6]):.5f}")

    del model1

    model2 = nn.Linear(in_channel, output_channel, bias=False)
    torch.nn.init.orthogonal_(model2.weight)
    with torch.no_grad():
        print(torch.diag(model2.weight @ model2.weight.t()))
    data2 = torch.randn(batch_size, in_channel)
    optim2 = CayleySGD(model2.parameters(), lr=0.1, momentum=0.9)
    for i in range(10):
        optim2.zero_grad()
        output = (model2(data2).abs()).mean()
        output.backward()
        optim2.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diag(model2.weight @ model2.weight.t()).sum().item():.2f}, "
                  f"weight: {torch.mean(model2.weight.flatten()[:6]):.5f}")

    model3 = nn.Linear(in_channel, output_channel, bias=False)
    torch.nn.init.orthogonal_(model3.weight)
    with torch.no_grad():
        print(torch.diag(model3.weight @ model3.weight.t()))
    data3 = torch.randn(batch_size, in_channel)
    optim3 = torch.optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)
    for i in range(10):
        optim3.zero_grad()
        output = (model3(data3).abs()).mean()
        output.backward()
        optim3.step()
        with torch.no_grad():
            print(f"step: {i}, loss: {output.item():.5f}, "
                  f"diag: {torch.diag(model3.weight @ model3.weight.t()).sum().item():.2f}, "
                  f"weight: {torch.mean(model3.weight.flatten()[:6]):.5f}")
