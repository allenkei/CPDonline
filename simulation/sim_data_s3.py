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


data_name = "s3"
num_time = 100
dim_d = args.dim_d
np.random.seed(42)
torch.manual_seed(42)
y_list = []


AR_param_1 = 0.8  
AR_param_2 = 0.2
mean_1 = torch.zeros(dim_d)  
mean_2 = 3.0 * torch.ones(dim_d)   
cov1 = torch.eye(dim_d)      
cov2 = torch.eye(dim_d)




if args.is_CP:

    num_seq = 50
    CP_time = 50

    for idx in range(num_seq):
      
      data = torch.zeros(num_time, dim_d)
      
      mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
      data_1 = mvn_1.sample((1,))  
      data[0] = data_1
      for t in range(1, CP_time):
        data[t] = AR_param_1 * data[t - 1] + (1 - AR_param_1) * mvn_1.sample((1,))

      mvn_2 = torch.distributions.MultivariateNormal(mean_2, cov2)
      data_2 = mvn_2.sample((1,)) 
      data[CP_time] = data_2

      for t in range(CP_time + 1, num_time):
        data[t] = AR_param_2 * data[t - 1] + (1 - AR_param_2) * mvn_2.sample((1,))

      y_list.append(data)

else:

    num_seq = 100

    for idx in range(num_seq):
  
        data = torch.zeros(num_time, dim_d)

        mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
        data_1 = mvn_1.sample((1,))  
        data[0] = data_1

        for t in range(1, num_time):
          data[t] = AR_param_1 * data[t - 1] + (1 - AR_param_1) * mvn_1.sample((1,))

        y_list.append(data)
  
  


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









