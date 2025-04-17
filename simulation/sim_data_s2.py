import argparse
import pandas as pd
import numpy as np
import torch


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dim_d', default=10, type=int)
  parser.add_argument('--is_CP')
  return parser.parse_args()
args = parse_args(); print(args)


data_name = "s2"
num_time = 100
dim_d = args.dim_d
np.random.seed(42)
torch.manual_seed(42)
y_list = []


def create_covariance(dim_d, correlation):
    A = torch.rand(dim_d, dim_d) - 1
    cov = A @ A.T
    cov += correlation * torch.eye(dim_d)
    return cov

alternating_mean = torch.zeros(dim_d)
alternating_mean[1::2] = 2.0
print("[INFO] alternating_mean:",alternating_mean)


mean_1 = torch.zeros(dim_d) 
mean_2 = alternating_mean #torch.ones(dim_d) * 2

cov1 = 0.5 * torch.eye(dim_d)
cov2 = create_covariance(dim_d, correlation=1.25)



if args.is_CP:

    num_seq = 50
    CP_time = 50

    for idx in range(num_seq):

        mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
        noise_1 = torch.rand(num_time - CP_time, dim_d) - 1
        data_1 = 0.5 * mvn_1.sample((CP_time,)) + 0.5 * noise_1


        mvn_2 = torch.distributions.MultivariateNormal(mean_2, cov2)
        t_dist_2 = torch.distributions.StudentT(5, torch.zeros(dim_d), torch.ones(dim_d)) 
        noise_2 = t_dist_2.sample((num_time - CP_time,))
        data_2 = 0.5 * mvn_2.sample((num_time - CP_time,)) + 0.5 * noise_2

        data = torch.cat((data_1, data_2), dim=0)
        y_list.append(data)

else:

    num_seq = 100

    for idx in range(num_seq):

        mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
        noise_1 = torch.rand(num_time , dim_d) - 1
        data_1 = 0.5 * mvn_1.sample((num_time,)) + 0.5 * noise_1

        
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









