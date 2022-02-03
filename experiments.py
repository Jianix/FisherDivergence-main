import os
import itertools
import sys


def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {
        'sigma_pert': [0.0, 0.01, 0.02, 0.04, 0.06],
        'mode': ['nn', 'sq_nn'],
        'spectral_norm': ['True', 'False'],
        'critic_n_hid': [30, 300],
        'l2_penalty': [1, 10],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py "
              "--log_freq 1000 "
              "--niters 1000 "
              "--MCs 10 "
              "--sigma_pert %s "
              "--mode %s "                  
              "--spectral_norm %s "
              "--critic_n_hid %s "
              "--l2_penalty %s "
              "--save %s "
              % (
                  temp['sigma_pert'],
                  temp['mode'],
                  temp['spectral_norm'],
                  temp['critic_n_hid'],
                  temp['l2_penalty'],
                  'taskid{}'.format(taskid))
              )


if __name__ == "__main__":
    main(sys.argv[1:])