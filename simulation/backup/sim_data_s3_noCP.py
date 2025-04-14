import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s3"
num_seq = 100
num_time = 100
dim_d = 10 # 10, 20



AR_param_1 = 0.8
mean_1 = torch.zeros(dim_d)    
cov1 = torch.eye(dim_d)      


np.random.seed(42)
torch.manual_seed(42)
y_list = []


for idx in range(num_seq):
  
  data = torch.zeros(num_time, dim_d)

  mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
  data_1 = mvn_1.sample((1,))  
  data[0] = data_1

  for t in range(1, num_time):
    data[t] = AR_param_1 * data[t - 1] + (1 - AR_param_1) * mvn_1.sample((1,))

  y_list.append(data)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_d{}_C.npz'.format(data_name, dim_d), y_list = y_list)
print('[INFO] data without CP saved')









