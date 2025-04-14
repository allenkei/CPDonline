import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s2"
num_seq = 100
num_time = 100
dim_d = 10 # 10, 20


np.random.seed(42)
torch.manual_seed(42)
y_list = []


df = 3 # t-dist
mean_1 = torch.zeros(dim_d) 
cov1 = 1.25 * torch.eye(dim_d)



#print('[INFO] mean_1:', mean_1)
#print('[INFO] mean_2:', mean_2)


for idx in range(num_seq):

    t_dist = torch.distributions.StudentT(df, torch.zeros(dim_d), torch.ones(dim_d)) 
    t_noise_1 = t_dist.sample((num_time,))

    mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
    data_1 = mvn_1.sample((num_time,)) + t_noise_1
    
    y_list.append(data_1)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_d{}_C.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data without CP saved')









