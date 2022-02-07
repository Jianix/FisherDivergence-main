import torch
import pandas as pd

import matplotlib.pyplot as plt

taskids = 36

df = pd.DataFrame(columns=["sigma_pert", "l2_penalty", "RBM_dim", "GoF_reject_rate", "id_reject_rate"])

# baseline_df = pd.DataFrame(columns=["sigma_pert", "l2_penalty"])

for taskid in range(taskids):
    try:
        result = torch.load('taskid{}/result.pt'.format(taskid), map_location=torch.device('cpu'))
        df.loc[len(df.index)] = list(result.values())

        # baseline_df.loc[len(baseline_df.index)] = [result['sigma_pert'], result['l2_penalty']]
    except:
        print('missing')

# TODO: finish this part, construct a table
df_sorted = df.sort_values('RBM_dim')

print(df_sorted)
