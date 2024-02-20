import math
import torch
from torch.optim.optimizer import Optimizer


class pbAdam2(Optimizer):
    r"""Implements pbAdam.
    This was inherited from 'pbSGD: Powered Stochastic Gradient Descent
    Methods for Accelerated Nonconvex Optimization'
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        gamma (float): Powerball function parameter (default: 1.0)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, gamma=1., betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, gamma=gamma, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(pbAdam2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pbAdam2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                # set_trace()
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step_t, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']

                # d_p = (torch.sign(p.grad) * torch.pow(torch.abs(p.grad), gamma)).data
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # update step
                step_t += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
                pb_exp_avg = torch.sign(exp_avg) * torch.pow(torch.abs(exp_avg), gamma)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p.conj(), value=1 - beta2)
                pb_exp_avg_sq = torch.sign(exp_avg_sq) * torch.pow(torch.abs(exp_avg_sq), gamma)

                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = group['lr'] / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, pb_exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (pb_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                p.data.addcdiv_(pb_exp_avg, denom, value=-step_size)

        return loss
