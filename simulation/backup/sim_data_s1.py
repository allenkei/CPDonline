import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

'''
import argparse
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name', default="s1")
  parser.add_argument('--dim_d', default=10) # 10, 20
  parser.add_argument('--num_seq', default=50)
  parser.add_argument('--num_time', default=100)
  parser.add_argument('--CP_time', default=50)
  return parser.parse_args()
'''


data_name = "s1"
num_seq = 50
num_time = 100
CP_time = 50
dim_d = 10 # 10 or 20


np.random.seed(42)
torch.manual_seed(42)
y_list = []

alternating_tensor = torch.zeros(dim_d)
alternating_tensor[1::2] = 3.0

# torch.cat(( torch.zeros(int(dim_d/2)), 3*torch.ones(int(dim_d/2)) )) 
# 3.0 * torch.ones(dim_d)

mean_1 = torch.zeros(dim_d) 
mean_2 = alternating_tensor

cov1 = torch.eye(dim_d) 
cov2 = torch.eye(dim_d) 

#print('[INFO] mean_1:', mean_1)
#print('[INFO] mean_2:', mean_2)


for idx in range(num_seq):

    mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
    mvn_2 = torch.distributions.MultivariateNormal(mean_2, cov2)
    
    data_1 = mvn_1.sample((CP_time,))
    data_2 = mvn_2.sample((num_time - CP_time,))

    data = torch.cat((data_1, data_2), dim=0)
    y_list.append(data)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_{}.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data saved')









