import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

def FSM(x, y, alpha, eps=1e-5):
    x_mu, x_var = torch.mean(x, dim=[2, 3, 4], keepdim=True), torch.var(x, dim=[2, 3, 4], keepdim=True)
    y_mu, y_var = torch.mean(y, dim=[2, 3, 4], keepdim=True), torch.var(y, dim=[2, 3, 4], keepdim=True)
    x_norm = (x - x_mu) / torch.sqrt(x_var + eps)
    x_fsm = x_norm * torch.sqrt(y_var + eps) + y_mu
    x_mix = alpha * x + (1 - alpha) * x_fsm
    return x_mix

class WGAN3D_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(WGAN3D_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.init_layer = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.ModuleList()
        self.pyramid_features = []
        i, csize, cndf = 3, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            self.main.append(
                nn.Sequential(
                    nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(cndf),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layer = nn.Sequential(
                nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False),
                nn.BatchNorm3d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.main.append(layer)
            i+=3
            cndf = cndf * 2
            csize = csize / 2

        self.final_layer = nn.Conv3d(cndf, 1, 4, 1, 0, bias=False)

    def forward(self, input, alpha=0.5):
        x = self.init_layer(input)

        #for i in range(len(self.main)):
        #    x = self.main[i](x)
        #    if i == 0:  # Assuming we want to apply FSMR after the first layer
        #        y = x[torch.randperm(x.size(0))]  # Shuffle batch to get 'y'
        #        x = FSM(x, y, alpha)

        for i in range(len(self.main)):
            x = self.main[i](x)

            # Apply FSM after each layer
            y = x[torch.randperm(x.size(0))]  # Shuffle batch to get 'y'
            x = FSM(x, y, alpha)

        x = self.final_layer(x)
        return x.view(-1, 1)

class LaplacianPyramidGenerator(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(LaplacianPyramidGenerator, self).__init__()
        self.ngpu = ngpu
        self.isize = isize
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.n_extra_layers = n_extra_layers

        self.init_layer = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True)
        )

        self.main = nn.ModuleList()
        csize, cndf = 4, ngf * 8

        while csize < isize // 2:
            self.main.append(nn.Sequential(
                nn.ConvTranspose3d(cndf, cndf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm3d(cndf // 2),
                nn.ReLU(True)
            ))
            cndf = cndf // 2
            csize = csize * 2

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(cndf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, level, input_img=None, target_size=None):
        if level == 0:
            x = self.init_layer(input)
            for i in range(len(self.main)):
                x = self.main[i](x)
            x = self.final_layer(x)
            return x
        else:
            x = self.init_layer(input)
            for i in range(len(self.main)):
                if i == 0:
                    input_img = F.interpolate(input_img, size=x.shape[2:], mode='trilinear', align_corners=False)
                    x = x + input_img
                x = self.main[i](x)
            residual = x - F.interpolate(input_img, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = self.final_layer(residual)
            return x

class WGAN3D_G_CPU(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(WGAN3D_G_CPU, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=True),
            nn.BatchNorm3d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=True))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)

class Wasserstein(nn.Module):
    def __init__(self, weight_clip_value, gradient_penalty_weight, netD):
        super(Wasserstein, self).__init__()
        self.weight_clip_value = weight_clip_value
        self.gradient_penalty_weight = gradient_penalty_weight
        self.netD = netD

    def forward(self, pred_real, pred_fake, real_samples, fake_samples):
        batch_size = real_samples.size(0)

        loss_real = -torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        loss = loss_real + loss_fake

        # Gradient penalty
        epsilon = torch.rand(batch_size, 1, 1, 1, 1).to(real_samples.device)
        x_hat = epsilon * real_samples.to(real_samples.device) + (1 - epsilon) * fake_samples.to(real_samples.device)
        x_hat.requires_grad = True
        self.netD.to(real_samples.device)
        pred_hat = self.netD(x_hat)

        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                        grad_outputs=torch.ones_like(pred_hat),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = self.gradient_penalty_weight * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

        loss += gradient_penalty

        return loss

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth*width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth*width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth*width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, height, width)
        out = self.gamma*out + x
        return out, attention

class WassersteinDiv(nn.Module):
    def __init__(self, gradient_penalty_weight, netD):
        super(WassersteinDiv, self).__init__()
        self.gradient_penalty_weight = gradient_penalty_weight
        self.netD = netD

    def forward(self, pred_real, pred_fake, real_samples):
        batch_size = real_samples.size(0)

        loss_real = -torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        loss = loss_real + loss_fake

        # Gradient penalty
        real_samples.requires_grad = True
        self.netD.to(real_samples.device)
        pred_real = self.netD(real_samples)

        gradients = torch.autograd.grad(outputs=pred_real, inputs=real_samples,
                                        grad_outputs=torch.ones_like(pred_real),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = self.gradient_penalty_weight * torch.mean(gradients.norm(2, dim=1) ** 2)

        loss += gradient_penalty

        return loss

class WassersteinLPO(nn.Module):
    def __init__(self, weight_clip_value, lipschitz_penalty_weight, netD):
        super(WassersteinLP, self).__init__()
        self.weight_clip_value = weight_clip_value
        self.lipschitz_penalty_weight = lipschitz_penalty_weight
        self.netD = netD

    def forward(self, pred_real, pred_fake, real_samples, fake_samples):
        batch_size = real_samples.size(0)

        loss_real = -torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        loss = loss_real + loss_fake

        # Lipschitz penalty
        epsilon = torch.rand(batch_size, 1, 1, 1, 1).to(real_samples.device)
        x_hat = epsilon * real_samples.to(real_samples.device) + (1 - epsilon) * fake_samples.to(real_samples.device)
        x_hat.requires_grad = True
        self.netD.to(real_samples.device)
        pred_hat = self.netD(x_hat)

        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                        grad_outputs=torch.ones_like(pred_hat),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        lipschitz_penalty = self.lipschitz_penalty_weight * torch.mean(gradients.norm(2, dim=1))

        loss += lipschitz_penalty

        # Weight clipping
        #for p in self.netD.parameters():
            #p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

        return loss

class WassersteinLP(nn.Module):
    def __init__(self, weight_clip_value, lipschitz_penalty_weight, netD, lambda_reg=10.0):
        super(WassersteinLP, self).__init__()
        self.weight_clip_value = weight_clip_value
        self.lipschitz_penalty_weight = lipschitz_penalty_weight
        self.netD = netD
        self.lambda_reg = lambda_reg

    def forward(self, pred_real, pred_fake, real_samples, fake_samples):
        batch_size = real_samples.size(0)

        # Original WGAN loss
        loss_real = -torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        loss = loss_real + loss_fake

        # Lipschitz penalty
        epsilon = torch.rand(batch_size, 1, 1, 1, 1).to(real_samples.device)
        x_hat = epsilon * real_samples.to(real_samples.device) + (1 - epsilon) * fake_samples.to(real_samples.device)
        x_hat.requires_grad = True
        self.netD.to(real_samples.device)
        pred_hat = self.netD(x_hat)

        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                        grad_outputs=torch.ones_like(pred_hat),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        lipschitz_penalty = self.lipschitz_penalty_weight * torch.mean(gradients.norm(2, dim=1))

        loss += lipschitz_penalty

        # FSM regularization
        y = real_samples[torch.randperm(real_samples.size(0))]  # Shuffle batch to get 'y'
        fsm_loss = torch.mean((real_samples - FSM(real_samples, y, 0.5)) ** 2)
        loss += self.lambda_reg * fsm_loss

        return loss

class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))

    def forward(self, x):
        mean_in = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
        var_in = torch.var(x, dim=[0, 2, 3, 4], keepdim=True, unbiased=False)

        mean_ln = torch.mean(x, dim=[2, 3, 4], keepdim=True)
        var_ln = torch.var(x, dim=[2, 3, 4], keepdim=True, unbiased=False)

        mean_weight = F.softmax(self.mean_weight, dim=0)
        var_weight = F.softmax(self.var_weight, dim=0)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
        var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (torch.sqrt(var + self.eps))
        x = x * self.weight + self.bias
        return x

class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FilterResponseNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3, 4], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        return torch.max(self.gamma * x + self.beta, self.tau)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim
import pdb
import math
import logging
import torch.distributed as dist


class DAdaptSGD(torch.optim.Optimizer):
    r"""
    Implements SGD with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        momentum (float):
            Momentum value in  the range [0,1) (default: 0).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int):
            Log using print every k steps, default 0 (no logging).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. More conservative values like 1.02 may
            help if training is unstable.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """

    def __init__(self, params,
                 lr=1.0,
                 momentum=0.0,
                 weight_decay=0,
                 log_every=0,
                 d0=1e-6, growth_rate=float('inf'),
                 fsdp_in_use=False):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay, k=0,
                        log_every=log_every,
                        numerator_weighted=0.0,
                        d=d0,
                        growth_rate=growth_rate,
                        fsdp_in_use=fsdp_in_use)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)

        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        numerator_weighted = group['numerator_weighted']
        growth_rate = group['growth_rate']
        d = group['d']
        fsdp_in_use = group['fsdp_in_use']

        group = self.param_groups[0]

        sk_sq = 0.0

        if k == 0:
            g_sq = 0.0
            for group in self.param_groups:
                group_lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if hasattr(p, "_fsdp_flattened"):
                        fsdp_in_use = True
                    grad = p.grad.data

                    # Apply weight decay
                    if decay != 0:
                        grad.add(p.data, alpha=decay)

                    state = self.state[p]

                    if group_lr > 0.0:
                        g_sq += (grad * grad).sum().item()

            if fsdp_in_use:
                dist_tensor = torch.zeros(1).cuda()
                dist_tensor[0] = g_sq
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_gsq = dist_tensor[0]
            else:
                global_gsq = g_sq
            group['g0_norm'] = g0_norm = math.sqrt(global_gsq)

        g0_norm = group['g0_norm']

        dlr = d * lr / g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                s = state['s']

                if group_lr > 0.0:
                    numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()

                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()
            ######

        d_hat = d

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = sk_sq
                dist_tensor[1] = numerator_weighted
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_sk_sq = dist_tensor[0]
                global_numerator_weighted = dist_tensor[1]
            else:
                global_sk_sq = sk_sq
                global_numerator_weighted = numerator_weighted

            d_hat = 2 * global_numerator_weighted / math.sqrt(global_sk_sq)
            d = max(d, min(d_hat, d * growth_rate))

        # if we have not done any updates
        # if we have any gradients available, will have sk_sq > 0 (unless \|g\|=0)
        if global_sk_sq == 0:
            return loss

        if log_every > 0 and k % log_every == 0:
            logging.info(
                f"(r={self.rank},k={k}) dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_norm={math.sqrt(global_sk_sq)} numerator_weighted={global_numerator_weighted} g0_norm={g0_norm}")

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            ######################################
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1 - ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss
