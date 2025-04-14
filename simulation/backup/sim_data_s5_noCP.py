import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s5"
num_seq = 200
num_time = 100
dim_d = 20 # 10, 20




np.random.seed(42)
torch.manual_seed(42)
y_list = []


def gen_ar_uniform_data(num_points, dimensions, phi=0.5):

    data = np.zeros((num_points, dimensions))
    data[0] = np.random.uniform(0, 1, dimensions)
    
    for t in range(1, num_points):
        noise = np.random.uniform(0, 1, dimensions)
        data[t] = phi * data[t-1] + noise
    
    return data



for idx in range(num_seq):
    
    before = gen_ar_uniform_data(num_time, dim_d)
    data = torch.tensor(before, dtype=torch.float32)
    
    y_list.append(data)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_d{}_C.npz'.format(data_name, dim_d), y_list = y_list)
print('[INFO] data without CP saved')









