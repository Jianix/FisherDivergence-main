import torch
import pandas as pd
import matplotlib.pyplot as plt

taskids = 27
MCs = 10

df = pd.DataFrame(columns=["sigma_pert", "l2_penalty", "RBM_dim", "GoF_reject_rate", "id_reject_rate"])

baseline_df = pd.DataFrame(columns=["sigma_pert",  "l2_penalty", "RBM_dim"])


for taskid in range(taskids):
    for mc in range(1,MCs+1):
        try:
            result = torch.load('taskid{}/result{}.pt'.format(taskid, mc), map_location=torch.device('cpu'))
            df.loc[len(df.index)] = list(result.values())
            baseline_df.loc[len(baseline_df.index)] = [result['sigma_pert'], result['l2_penalty'], result['RBM_dim']]
        except:
            print('missing')


# TODO: finish this part
# method
methods_list = [['0.1_50,40', '0.1_100,80', '0.1_200,100'],

]

for methods in methods_list:

    temp = df.loc[df['method'].isin(methods)]

    i = 1
    for metric in ["GoF_reject_rate", "id_reject_rate"]:

        true = baseline_df.groupby(['sigma_pert', 'l2_penalty'])[metric].mean()
        df2 = temp.groupby(['sigma_pert', 'method'])[metric].mean()
        df3 = pd.concat([true.unstack(), df2.unstack()], axis=1)

        print(df3)
        plt.subplot(1, 2, i)
        df3.plot(legend=False, ax=plt.gca(), yerr=temp.groupby(['sigma_pert', 'l2_penalty'])[metric].std().unstack())
        plt.title('{}'.format(metric))
        i += 1


    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
