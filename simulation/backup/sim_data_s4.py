import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s4"
num_seq = 50
num_time = 100
CP_time = 50
dim_d = 10




np.random.seed(42)
torch.manual_seed(42)
y_list = []


def gen_uniform_data(num_points, dimensions):
    return np.random.uniform(0, 1, size=(num_points, dimensions))

def gen_beta_data(num_points, dimensions, alpha=2.0, beta=5.0):
    return np.random.beta(alpha, beta, size=(num_points, dimensions))

for idx in range(num_seq):
    data = torch.zeros(num_time, dim_d)
    
    before = gen_uniform_data(CP_time, dim_d)
    data[:CP_time] = torch.tensor(before, dtype=torch.float32)
    
    after = gen_beta_data(num_time - CP_time, dim_d)
    data[CP_time:] = torch.tensor(after, dtype=torch.float32)

    y_list.append(data)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_{}.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data saved')









