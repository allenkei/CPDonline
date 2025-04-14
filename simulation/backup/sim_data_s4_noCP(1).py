import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s4"
num_seq = 300
num_time = 100
dim_d = 10



np.random.seed(42)
torch.manual_seed(42)
y_list = []


def gen_uniform_data(num_points, dimensions):
    return np.random.uniform(0, 1, size=(num_points, dimensions))

for idx in range(num_seq):
    
    before = gen_uniform_data(num_time, dim_d)
    data = torch.tensor(before, dtype=torch.float32)
    
    y_list.append(data)


y_list = np.array(y_list)
#print(y_list[0])


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
np.savez('data/data_{}_{}_C.npz'.format(data_name,dim_d), y_list = y_list)
print('[INFO] data without CP saved')









