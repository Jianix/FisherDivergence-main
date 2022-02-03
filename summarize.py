import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

taskids = 80
MCs = 10

df = pd.DataFrame(columns=["sigma_pert", "method", "mc",
                           "FDtrue_train", "FDhat_train", "Aq_f_train", "Aq_f_z_train", "Ap_f_train", "Ap_f_z_train",
                           "FDtrue_test", "FDhat_test", "Aq_f_test", "Aq_f_z_test", "Ap_f_test", "Ap_f_z_test"])

baseline_df = pd.DataFrame(columns=["sigma_pert", "method", "FDhat_train", "FDhat_test"])

for taskid in range(taskids):
    for mc in range(1,MCs+1):
        try:
            result = torch.load('taskid{}/result{}.pt'.format(taskid, mc), map_location=torch.device('cpu'))
            df.loc[len(df.index)] = list(result.values())
            baseline_df.loc[len(baseline_df.index)] = [result['sigma_pert'], 'truth', result['FDtrue_train'], result['FDtrue_test']]
        except:
            print('missing')


# just look at a subset of the results
methods_list = [
    ['nn_True_300_1.0', 'nn_True_30_1.0', 'nn_True_300_10.0', 'nn_True_30_10.0', 'sq_nn_True_300_1.0', 'sq_nn_True_30_1.0', 'sq_nn_True_300_10.0', 'sq_nn_True_30_10.0'],
    ['sq_nn_True_300_10.0', 'sq_nn_True_30_10.0', 'nn_True_30_10.0'],
    ['nn_False_300_1.0', 'nn_False_30_1.0', 'nn_False_300_10.0', 'nn_False_30_10.0', 'sq_nn_False_300_1.0', 'sq_nn_False_30_1.0', 'sq_nn_False_300_10.0', 'sq_nn_False_30_10.0'],
    ['sq_nn_False_300_10.0', 'sq_nn_False_30_10.0', 'nn_False_30_10.0'],
]

for methods in methods_list:

    temp = df.loc[df['method'].isin(methods)]

    i = 1
    for metric in ['FDhat_train', 'FDhat_test']:

        true = baseline_df.groupby(['sigma_pert', 'method'])[metric].mean()
        df2 = temp.groupby(['sigma_pert', 'method'])[metric].mean()
        df3 = pd.concat([true.unstack(), df2.unstack()], axis=1)

        print(df3)
        plt.subplot(1, 2, i)
        df3.plot(legend=False, ax=plt.gca(), yerr=temp.groupby(['sigma_pert', 'method'])[metric].std().unstack())
        plt.title('{}'.format(metric))
        i += 1

    # ax = plt.gca()
    # ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
