import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import networks
import argparse
import os
import matplotlib

matplotlib.use('Agg')
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm import tqdm


def try_make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


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
        unden = (x * b).sum(1) - .5 * (x ** 2).sum(1)  # + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc / 2.).sum(
            1)  # (xBc.exp() + (-xBc).exp()).log().sum(1)
        print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in tqdm(range(self.burn_in)):
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

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class Laplace(nn.Module):
    def __init__(self, mu, std):
        super(Laplace, self).__init__()
        self.dist = distributions.Laplace(mu, std)

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


def sample_batch(data, batch_size):
    all_inds = list(range(data.size(0)))
    chosen_inds = np.random.choice(all_inds, batch_size, replace=False)
    chosen_inds = torch.from_numpy(chosen_inds)
    return data[chosen_inds]


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input,
                               grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


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


class SpectralLinear(nn.Module):
    def __init__(self, n_in, n_out, max_sigma=1.):
        super(SpectralLinear, self).__init__()
        self.linear = spectral_norm(nn.Linear(n_in, n_out))
        self.scale = nn.Parameter(torch.zeros((1,)))
        self.max_sigma = max_sigma

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.scale) * self.max_sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['gaussian-laplace', 'laplace-gaussian',
                                           'gaussian-pert', 'rbm-pert', 'rbm-pert1'], type=str)

    # add the perturbation level as input parameter, default = 0.02
    parser.add_argument('--sigma_pert', type=float, default=.02)

    # the lambda penalty term
    parser.add_argument('--l2', type=float, default=0.)

    # change the way we input the dimensions of RBM to suit the experiment purpose
    parser.add_argument('--RBM_dim', type=int, nargs=2)

    #parser.add_argument('--dim_x', type=int, default=50)
    #parser.add_argument('--dim_h', type=int, default=40)



    parser.add_argument('--maximize_power', action="store_true")
    parser.add_argument('--maximize_adj_mean', action="store_true")
    parser.add_argument('--val_power', action="store_true")
    parser.add_argument('--val_adj_mean', action="store_true")
    #parser.add_argument('--dropout', action="store_true")


    parser.add_argument('--alpha', type=float, default=.05)

    parser.add_argument('--save', type=str, default='/tmp/test_ksd')

    #parser.add_argument('--test_type', type=str, default='mine')

    #parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--num_const', type=float, default=1e-6)

    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)

    # TODO: go through where these arguments are been used
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--test_burn_in', type=int, default=0)
    #parser.add_argument('--mode', type=str, default="fs")  # TODO: check args.mode
    #parser.add_argument('--viz_freq', type=int, default=100)
    #parser.add_argument('--save_freq', type=int, default=10000)

    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--base_dist', action="store_true")

    #parser.add_argument('--t_iters', type=int, default=5)
    #parser.add_argument('--k_dim', type=int, default=1)
    #parser.add_argument('--sn', type=float, default=-1.)

    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--quadratic', action="store_true")
    parser.add_argument('--n_steps', type=int, default=100)
    #parser.add_argument('--both_scaled', action="store_true")

    # the number of iterations for each pair of values of lambda and perturbation rate
    # this is used as the denominator for calculating the rejection rate
    # when varying the lambda and perturb rate
    parser.add_argument('--n_rej_iter', type=int, default=100)

    args = parser.parse_args()
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


    # set seed for the reproduction purpose
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.test == "gaussian-laplace":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        p_dist = Gaussian(mu, std)
        q_dist = Laplace(mu, std)
    elif args.test == "laplace-gaussian":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        q_dist = Gaussian(mu, std)
        p_dist = Laplace(mu, std / (2 ** .5))
    elif args.test == "gaussian-pert":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        p_dist = Gaussian(mu, std)
        q_dist = Gaussian(mu + torch.randn_like(mu) * args.sigma_pert, std)
    elif args.test == "rbm-pert1":
        B = randb((args.dim_x, args.dim_h)) * 2. - 1.
        c = torch.randn((1, args.dim_h))
        b = torch.randn((1, args.dim_x))

        p_dist = GaussianBernoulliRBM(B, b, c)
        B2 = B.clone()
        B2[0, 0] += torch.randn_like(B2[0, 0]) * args.sigma_pert
        q_dist = GaussianBernoulliRBM(B2, b, c)
    else:  # args.test == "rbm-pert"
        B = randb((args.dim_x, args.dim_h)) * 2. - 1.
        c = torch.randn((1, args.dim_h))
        b = torch.randn((1, args.dim_x))

        p_dist = GaussianBernoulliRBM(B, b, c)
        q_dist = GaussianBernoulliRBM(B + torch.randn_like(B) * args.sigma_pert, b, c)




    import numpy as np
    data = p_dist.sample(args.n_train + args.n_val + args.n_test).detach()
    data_train = data[:args.n_train]
    data_rest = data[args.n_train:]
    data_val = data_rest[:args.n_val].requires_grad_()
    data_test = data_rest[args.n_val:].requires_grad_()
    assert data_test.size(0) == args.n_test

    critic = networks.SmallMLP(args.dim_x, n_out=args.dim_x, n_hid=300, dropout=args.dropout)
    optimizer = optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def stein_discrepency(x, exact=False, sq_flag = True):
        if "rbm" in args.test:
            sq = q_dist.score_function(x)
            sp = p_dist.score_function(x)
        else:
            logq_u = q_dist(x)
            sq = keep_grad(logq_u.sum(), x)
        fx = critic(x)
        if args.dim_x == 1:
            fx = fx[:, None]

        if sq_flag:
            sq_fx = (sq * fx).sum(-1)
        else:
            sq_fx = (sp * fx).sum(-1)

        if exact:
            tr_dfdx = exact_jacobian_trace(fx, x)
        else:
            tr_dfdx = approx_jacobian_trace(fx, x)

        norms = (fx * fx).sum(1)
        stats = (sq_fx + tr_dfdx)
        return stats, norms

    # training phase
    best_val = -np.inf
    validation_metrics = []
    test_statistics = []
    critic.train()

    # add the reject accumulation variable here to count the number of times with rejecting the null
    reject_times: int = 0

    # add the reject accumulation variable for testing equation 3
    reject_id_times : int = 0

    for i in range(args.n_rej_iter):

        for itr in range(args.n_iters):
            optimizer.zero_grad()
            x = sample_batch(data_train, args.batch_size)
            x = x.to(device)
            x.requires_grad_()

            stats, norms = stein_discrepency(x)
            mean, std = stats.mean(), stats.std()
            l2_penalty = norms.mean() * args.l2

            if args.maximize_power:
                loss = -1. * mean / (std + args.num_const) + l2_penalty
            elif args.maximize_adj_mean:
                loss = -1. * mean + std + l2_penalty
            else:
                # go to this branch by default if not put in any argument
                loss = -1. * mean + l2_penalty

            loss.backward()
            optimizer.step()

            if itr % args.log_freq == 0:
                print("Iter {}, Loss = {}, Mean = {}, STD = {}, L2 {}".format(itr,
                                                                              loss.item(), mean.item(), std.item(),
                                                                              l2_penalty.item()))

            if itr % args.val_freq == 0:
                critic.eval()
                val_stats, _ = stein_discrepency(data_val, exact=True)
                test_stats, _ = stein_discrepency(data_test, exact=True)
                print("Val: {} +/- {}".format(val_stats.mean().item(), val_stats.std().item()))
                print("Test: {} +/- {}".format(test_stats.mean().item(), test_stats.std().item()))

                if args.val_power:
                    validation_metric = val_stats.mean() / (val_stats.std() + args.num_const)
                elif args.val_adj_mean:
                    validation_metric = val_stats.mean() - val_stats.std()
                else:
                    validation_metric = val_stats.mean()

                test_statistic = test_stats.mean() / (test_stats.std() + args.num_const)

                if validation_metric > best_val:
                    print("Iter {}, Validation Metric = {} > {}, Test Statistic = {}, Current Best!".format(itr,
                                                                                                            validation_metric.item(),
                                                                                                            best_val,
                                                                                                            test_statistic.item()))
                    best_val = validation_metric.item()
                else:
                    print("Iter {}, Validation Metric = {}, Test Statistic = {}, Not best {}".format(itr,
                                                                                                     validation_metric.item(),
                                                                                                     test_statistic.item(),
                                                                                                     best_val))
                validation_metrics.append(validation_metric.item())
                test_statistics.append(test_statistic)
                critic.train()

        # TODO: evaluate the equation 3 here
        id_stats, _ = stein_discrepency(data_test, sq_flag=False)
        id_test_statistic = id_stats.mean() / (id_stats.std() + args.num_const) * (args.n_test)**0.5
        print("The test statistic for validating equation 3 is {}".format(id_test_statistic))
        id_threshold = distributions.Normal(0, 1).icdf(torch.ones((1,)) * (1. - args.alpha)).item()
        if (id_test_statistic > id_threshold) or (id_test_statistic < (-1)*id_threshold):
            # then by the t-test our null hypothesis (equation 3) is rejected
            reject_id_times += 1



        best_ind = np.argmax(validation_metrics)
        best_test = test_statistics[best_ind]

        print("Best val is {}, best test is {}".format(best_val, best_test))
        test_stat = best_test * args.n_test ** .5
        threshold = distributions.Normal(0, 1).icdf(torch.ones((1,)) * (1. - args.alpha)).item()

        if (test_stat > threshold) or (test_stat < (-1)*threshold):
            # reject accumulation variable +1
            reject_times += 1
            print("Now the total number of rejection is {}".format(reject_times))


        try_make_dirs(os.path.dirname(args.save))
        with open(args.save, 'w') as f:
            f.write(str(test_stat) + '\n')
            if (test_stat > threshold) or (test_stat < (-1)*threshold):
                print("{} > {}, rejct Null".format(test_stat, threshold))
                f.write("reject")
            else:
                print("{} <= {}, accept Null".format(test_stat, threshold))
                f.write("accept")


        # reset the lists and best value, have the critic and optimizer reseted as well
        best_val = -np.inf
        validation_metrics.clear()
        test_statistics.clear()

        # resample the data each time in order to catch the sampling variation
        data = p_dist.sample(args.n_train + args.n_val + args.n_test).detach()
        data_train = data[:args.n_train]
        data_rest = data[args.n_train:]
        data_val = data_rest[:args.n_val].requires_grad_()
        data_test = data_rest[args.n_val:].requires_grad_()
        assert data_test.size(0) == args.n_test

        # train the critic again with the new data sample
        critic = networks.SmallMLP(args.dim_x, n_out=args.dim_x, n_hid=300, dropout=args.dropout)
        optimizer = optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic.train()




    # compute the rejection rate here using reject_times / total_number_experiments
    reject_rate = reject_times/args.n_rej_iter
    reject_id_rate = reject_id_times/args.n_rej_iter
    #print("{} experiments have been run. "
    #      "The current rejection rate for goodness-of-fit test is {}. "
    #      "And the rejection rate for validating equation 3 is {}.".format(args.n_rej_iter, reject_rate,reject_id_rate))

    result = dict()
    result.update({
        "sigma_pert": args.sigma_pert,
        "l2_penalty": args.l2_penalty,
        "RBM_dim": args.RBM_dim,
        "GoF_reject_rate": reject_rate,
        "identity_reject_rate": reject_id_rate
    })

    print(result)
    torch.save(result, '{}/result.pt'.format(args.save))


if __name__ == "__main__":
    main()
