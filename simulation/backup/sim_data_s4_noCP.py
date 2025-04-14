import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s4"
num_seq = 100
num_time = 100
dim_d = 10 # 10, 20


np.random.seed(42)
torch.manual_seed(42)
y_list = []


cov1 = 0.5 * torch.eye(dim_d) 

for idx in range(num_seq):

    data_1 = []
    for t in range(num_time):
        bernoulli_sample = torch.bernoulli(torch.tensor(0.5))

        if bernoulli_sample == 0:
            mean_1 = -2 * torch.ones(dim_d)
        else:
            mean_1 =  2 * torch.ones(dim_d)

        mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
        data_1.append(mvn_1.sample())

    data = torch.stack(data_1)
    y_list.append(data_1)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_d{}_C.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data without CP saved')









