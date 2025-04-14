import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


data_name = "s1"
num_seq = 300
num_time = 100
dim_d = 10


np.random.seed(42)
torch.manual_seed(42)
y_list = []

mean_1 = torch.zeros(dim_d) 
cov1 = torch.eye(dim_d) 
#print('[INFO] mean_1:', mean_1)


for idx in range(num_seq):

    mvn_1 = torch.distributions.MultivariateNormal(mean_1, cov1)
    data_1 = mvn_1.sample((num_time,))
    y_list.append(data_1)


y_list = np.array(y_list)


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_{}_C.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data without CP saved')









