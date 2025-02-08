import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s1"
num_seq = 20
num_time = 100 
dim_d = 10




np.random.seed(42)
y_list = []

mean_1 = torch.zeros(dim_d) - 1.0
mean_2 = torch.zeros(dim_d) + 5.0 
print('[INFO] mean_1:', mean_1)
print('[INFO] mean_2:', mean_2)

cov1 = torch.eye(dim_d)*0.5
cov2 = torch.eye(dim_d)*1.5


for idx in range(num_seq):

    mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
    mvn_2 = torch.distributions.MultivariateNormal(mean_2, cov2)
    
    data_1 = mvn_1.sample((num_time // 2,))
    data_2 = mvn_2.sample((num_time // 2,))
    data = torch.cat((data_1, data_2), dim=0)
    
    y_list.append(data)


y_list = np.array(y_list)

#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}.npz'.format(data_name), y_list = y_list)
print('[INFO] data saved')









