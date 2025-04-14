import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data_name = "s1"
num_seq = 100
num_time = 100
dim_d = 10 # 10, 20


np.random.seed(42)
torch.manual_seed(42)
y_list = []



alpha_pareto = 2.0
scale_pareto = 1.0




for idx in range(num_seq):

    data_1 = (np.random.pareto(alpha_pareto, size=(num_time, dim_d)) + 1) * scale_pareto
    data_1_noise = np.random.uniform(-1, 1, size=(num_time, dim_d))
    data_1 -= data_1_noise
    
    y_list.append(data_1)


y_list = np.array(y_list)


print("[INFO] y_list.shape:",y_list.shape)

# Save to an .npz file
y_list_tensor = torch.tensor(y_list, dtype=torch.float32)
np.savez('data/data_{}_d{}_C.npz'.format(data_name, dim_d), y_list = y_list_tensor)
print('[INFO] data without CP saved')









