import argparse
import os
import matplotlib
matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim

import numpy as np
import networks
import utils
import pickle
import random

def try_make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()


class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        # self.B = B
        # self.b = b
        # self.c = c
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in

    def score_function(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        # print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden2 # TODO: unden or unden2

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in range(self.burn_in):
            x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super(Gaussian, self).__init__()
        self.dist = distributions.Normal(mu, std)

    # TODO: finish score function
    # def score_function(self, x):  # dlogp(x)/dx

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class Laplace(nn.Module):
    def __init__(self, mu, std):
        super(Laplace, self).__init__()
        self.dist = distributions.Laplace(mu, std)

    # TODO: finish score function
    # def score_function(self, x):  # dlogp(x)/dx

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class EBM(nn.Module):
    def __init__(self, net, base_dist=None):
        super(EBM, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc)
            self.base_logstd = nn.Parameter(base_dist.scale.log())
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        return self.net(x) + bd # looks like this is = log unnormalized density, self.net is -E_\theta(x)


def sample_batch(data, batch_size):
    all_inds = list(range(data.size(0)))
    chosen_inds = np.random.choice(all_inds, batch_size, replace=False)
    chosen_inds = torch.from_numpy(chosen_inds)
    return data[chosen_inds]


def approx_jacobian_trace(fx, x):

    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)

    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)


def calculate_fx(mode, sq, critic, l2_penalty, x):
    nnx = critic(x)
    if mode == 'nn':
        fx = nnx
    elif mode == 'sq_nn':
        fx = (sq - nnx)/(2*l2_penalty)
    elif mode == 'sq_gradnn':
        log_ptilde = nnx
        sp = keep_grad(log_ptilde.sum(), x)
        fx = (sq - sp)/(2*l2_penalty)
    fx_l2 = (fx * fx).sum(1).mean()
    nn_l2 = (nnx * nnx).sum(1).mean()
    return fx, fx_l2, nn_l2


def calculate_tr_dfdx(fx, x, exact_trace=False):
    # compute/estimate Tr(df/dx)
    if exact_trace:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = approx_jacobian_trace(fx, x)

    return tr_dfdx


def evaluate(fx, name, evaluation_data, args, z_test_flag=False):

    print('---------')

    evaluation_data.requires_grad_()
    sq = args.q_dist.score_function(evaluation_data)
    sp = args.p_dist.score_function(evaluation_data)
    
    tr_dfdx = calculate_tr_dfdx(fx, evaluation_data, exact_trace=True)

    # E_p A_q f
    sq_fx = (sq * fx).sum(-1)
    tr_Aq_f = (sq_fx + tr_dfdx).detach()
    FDhat = tr_Aq_f.mean() - args.l2_penalty * (fx * fx).sum(1).mean()
    if z_test_flag:
        z_score = tr_Aq_f.mean() / tr_Aq_f.std() * (tr_Aq_f.shape[0] ** .5)
        print('{} | E_x~p tr[ (A_q f*)(x) {}] | z-score = {}'
                    .format(name, tr_Aq_f.mean(), z_score))
    else:
        z_score = None

    print('{} | E_x~p tr[ (A_q f*)(x) ] - \lambda E_x~p ||f*(x)||^2_2 = {}'.format(name, FDhat))

    # E_p A_p f
    sp_fx = (sp * fx).sum(-1)
    tr_Ap_f = (sp_fx + tr_dfdx).detach()
    if z_test_flag:
        stein_z_score = tr_Ap_f.mean() / tr_Ap_f.std() * (tr_Ap_f.shape[0] ** .5)
        print('{} | E_x~p tr[ (A_p f*)(x) ] = {} | z-score {} '.format(name, tr_Ap_f.mean(), stein_z_score))
    else:
        stein_z_score = None
        print('{} | E_x~p tr[ (A_p f*)(x) ] = {}'.format(name, tr_Ap_f.mean()))
        
    return FDhat.detach(), tr_Aq_f.mean(), z_score, tr_Ap_f.mean(), stein_z_score


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--MCs', type=int, default=1)
    parser.add_argument('--data', choices=['gaussian-laplace', 'laplace-gaussian',
                                           'gaussian-pert', 'rbm-pert', 'rbm-pert1'], type=str, default='rbm-pert')
    parser.add_argument('--dim_x', type=int, default=50)
    parser.add_argument('--dim_h', type=int, default=40)
    parser.add_argument('--sigma_pert', type=float, default=.02)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)

    # training parameters
    parser.add_argument('--niters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--critic_weight_decay', type=float, default=0)
    parser.add_argument('--l2_penalty', type=float, default=1.0)
    parser.add_argument('--enforce_stein', action="store_true")

    # NN parameters
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'sq_nn', 'sq_gradnn'])
    parser.add_argument('--critic_n_hid', type=int, default=300)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--spectral_norm', type=str, default='True')

    # saving and logging
    parser.add_argument('--save', type=str, default='temp')
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)

    args = parser.parse_args()
    args.spectral_norm = (args.spectral_norm == 'True')
    args.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    try_make_dirs(args.save)

    print(args)

    for mc in range(1, args.MCs+1):

        torch.manual_seed(mc)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(mc)
        random.seed(mc)
        np.random.seed(mc)

        if args.data == "gaussian-laplace":
            mu = torch.zeros((args.dim_x,))
            std = torch.ones((args.dim_x,))
            p_dist = Gaussian(mu, std)
            q_dist = Laplace(mu, std)
        elif args.data == "laplace-gaussian":
            mu = torch.zeros((args.dim_x,))
            std = torch.ones((args.dim_x,))
            q_dist = Gaussian(mu, std)
            p_dist = Laplace(mu, std / (2 ** .5))
        elif args.data == "gaussian-pert":
            mu = torch.zeros((args.dim_x,))
            std = torch.ones((args.dim_x,))
            p_dist = Gaussian(mu, std)
            q_dist = Gaussian(mu + torch.randn_like(mu) * args.sigma_pert, std)
        elif args.data == "rbm-pert1":
            B = randb((args.dim_x, args.dim_h)) * 2. - 1.
            c = torch.randn((1, args.dim_h))
            b = torch.randn((1, args.dim_x))

            p_dist = GaussianBernoulliRBM(B, b, c)
            B2 = B.clone()
            B2[0, 0] += torch.randn_like(B2[0, 0]) * args.sigma_pert
            q_dist = GaussianBernoulliRBM(B2, b, c)
        elif args.data == "rbm-pert":
            B = randb((args.dim_x, args.dim_h)) * 2. - 1.
            c = torch.randn((1, args.dim_h))
            b = torch.randn((1, args.dim_x))

            p_dist = GaussianBernoulliRBM(B, b, c)
            q_dist = GaussianBernoulliRBM(B + torch.randn_like(B) * args.sigma_pert, b, c)

        args.p_dist = p_dist.to(args.device)
        args.q_dist = q_dist.to(args.device)

        # samples from p
        data = p_dist.sample(args.n_train + args.n_val + args.n_test).detach()
        data_train = data[:args.n_train]
        data_rest = data[args.n_train:]
        data_val = data_rest[:args.n_val].requires_grad_()
        data_test = data_rest[args.n_val:].requires_grad_()
        assert data_test.size(0) == args.n_test

        if args.mode == 'nn' or args.mode == 'sq_nn':
            critic = networks.SmallMLP(args.dim_x, n_out=args.dim_x, n_hid=args.critic_n_hid, dropout=False, spectral_norm=args.spectral_norm)  # modeling f(x)
        elif args.mode == 'sq_gradnn':
            critic = EBM(networks.SmallMLP(args.dim_x, n_hid=args.critic_n_hid, dropout=False), None)  # modeling ebm p
        critic.to(args.device)

        critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr, betas=(.5, .9),
                                      weight_decay=args.critic_weight_decay)

        time_meter = utils.RunningAverageMeter(0.98)
        end = time.time()

        for itr in range(args.niters):

            critic.train()
            critic_optimizer.zero_grad()

            x = sample_batch(data_train, args.batch_size)
            x = x.to(args.device)
            x.requires_grad_()

            # TODO: why is above different from keep_grad(q_dist(x).sum(), x)
            sq = q_dist.score_function(x)
            fx, fx_l2, nn_l2 = calculate_fx(args.mode, sq, critic, args.l2_penalty, x)
            tr_dfdx = calculate_tr_dfdx(fx, x)
            tr_Aq_f = (sq * fx).sum(-1) + tr_dfdx

            if args.enforce_stein:
                sp = p_dist.score_function(x)
                sp_fx = (sp * fx).sum(-1)
                tr_Ap_f = (sp_fx + tr_dfdx)
                adversarial_loss = -1. * tr_Aq_f.mean() + fx_l2 * args.l2_penalty + torch.max(tr_Ap_f.mean(),torch.tensor([0]))
            else:
                adversarial_loss = -1. * tr_Aq_f.mean() + fx_l2 * args.l2_penalty

            adversarial_loss.backward()
            critic_optimizer.step()

            time_meter.update(time.time() - end)

            if (itr + 1) % args.log_freq == 0:

                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | adversarial loss {:.4f} = - tr_Aq_f.mean() {} + penalty {} * critic L2 {:.4f} | nn L2 {:.4f} '.format(
                        itr, time_meter.val, time_meter.avg, adversarial_loss.item(), tr_Aq_f.mean(), args.l2_penalty, fx_l2, nn_l2)
                )
                critic.eval()

                data_train.requires_grad_()
                sq = q_dist.score_function(data_train)
                fx, fx_l2, _ = calculate_fx(args.mode, sq, critic, args.l2_penalty, data_train)
                evaluate(fx, 'train', data_train, args)

                sq = q_dist.score_function(data_test)
                fx, fx_l2, _ = calculate_fx(args.mode, sq, critic, args.l2_penalty, data_test)
                evaluate(fx, 'test', data_test, args, z_test_flag=True)

                print(log_message)

            end = time.time()

        result = dict()
        method = '{}_{}_{}_{}'.format(
            args.mode,
            args.spectral_norm,
            args.critic_n_hid,
            args.l2_penalty
        )
        result.update({
            "sigma_pert": args.sigma_pert,
            "method": method,
            "mc": mc
        })


        for name in ['train', 'test']:

            if name == 'train':
                evaluation_data = data_train
                evaluation_data.requires_grad_()
            elif name == 'test':
                evaluation_data = data_test

            print('MC{}-----{}----'.format(mc, name))
            # ground truth
            sq = args.q_dist.score_function(evaluation_data)
            sp = args.p_dist.score_function(evaluation_data)
            FDtrue = ((sq - sp) * (sq - sp)).sum(1).mean() / (4 * args.l2_penalty)
            print('True Fisher divergence: (1/4\lambda) E_x~p ||s_q - s_p ||_2^2 = {} '.format(FDtrue))

            ## estimated f
            fx, fx_l2, _ = calculate_fx(args.mode, sq, critic, args.l2_penalty, evaluation_data)
            FDhat, Aq_f, Aq_f_z, Ap_f, Ap_f_z = evaluate(fx, name, evaluation_data, args, z_test_flag=True if name=='test' else False)

            # print('-----{}----'.format(name))
            # ## fstar = (1/2\lambda) (s_q - s_p), the argmax to regularized Stein
            # fstar = (sq-sp)/(2*args.l2_penalty)
            # evaluate(fstar, 'true', evaluation_data, args, zscore=True if name=='test' else False)

            result.update({
                "FDtrue_{}".format(name): FDtrue.detach(),
                "FDhat_{}".format(name): FDhat,
                "Aq_f_{}".format(name): Aq_f,
                "Aq_f_z_{}".format(name): Aq_f_z,
                "Ap_f_{}".format(name): Ap_f,
                "Ap_f_z_{}".format(name): Ap_f_z,
            })

        print(result)
        torch.save(result, '{}/result{}.pt'.format(args.save, mc))


if __name__ == "__main__":
    main()