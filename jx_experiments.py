import os
import itertools
import sys


def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {
        'sigma_pert': [0.00, 0.02, 0.04, 0.06],
        'l2_penalty': [0.1, 1, 10],
        'RBM_dim': ["50 40", "100 80", "200 100"],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py "
              "--test rbm-pert "
              "--log_freq 1000 "
              "--n_iters 1000 "
              "--seed 940621 "
              "--sigma_pert %s "
              "--l2 %s "
              "--RBM_dim %s "
              "--save %s "
              % (
                  temp['sigma_pert'],
                  temp['l2_penalty'],
                  temp['RBM_dim'],
                  'taskid{}'.format(taskid))
              )


if __name__ == "__main__":
    main(sys.argv[1:])