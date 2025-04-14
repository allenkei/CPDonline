import argparse
import pandas as pd
import numpy as np
from pandas.core.arrays import boolean
import torch


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dim_d', default=10, type=int)
  parser.add_argument('--is_CP')
  return parser.parse_args()
args = parse_args(); print(args)


data_name = "s1"
num_time = 100
dim_d = args.dim_d
np.random.seed(42)
torch.manual_seed(42)
y_list = []


scale_exp = 1.0

mu_ln = 1.0
sigma_ln = 1.0



if args.is_CP:

    num_seq = 50
    CP_time = 50

    for idx in range(num_seq):

        data_1 = np.random.exponential(scale=scale_exp, size=(CP_time, dim_d))
        data_1_noise = np.random.uniform(0, 1, size=(CP_time, dim_d))
        data_1 += data_1_noise

        data_2 = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=(num_time - CP_time, dim_d))
        data_2_noise = np.random.uniform(0, 1, size=(num_time - CP_time, dim_d))
        data_2 += data_2_noise
        
        data = np.concatenate((data_1, data_2), axis=0)
        y_list.append(data)

else:

    num_seq = 100

    for idx in range(num_seq):

        data_1 = np.random.exponential(scale=scale_exp, size=(num_time, dim_d))
        data_1_noise = np.random.uniform(0, 1, size=(num_time, dim_d))
        data_1 += data_1_noise

        y_list.append(data_1)






y_list = np.array(y_list) 
print("[INFO] before:",np.mean(y_list[0][:50, :], axis=0))
print("[INFO] after:",np.mean(y_list[0][50:, :], axis=0))
print("[INFO] y_list.shape:",y_list.shape)



# Save to an .npz file
if args.is_CP:
    y_list_tensor = torch.tensor(y_list, dtype=torch.float32)
    np.savez('data/data_{}_d{}.npz'.format(data_name, dim_d), y_list = y_list_tensor)
    print('[INFO] data saved')
else:
    y_list_tensor = torch.tensor(y_list, dtype=torch.float32)
    np.savez('data/data_{}_d{}_C.npz'.format(data_name, dim_d), y_list = y_list_tensor)
    print('[INFO] data without CP saved')









