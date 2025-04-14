import argparse
import pandas as pd
import numpy as np
import torch


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dim_d', default=10, type=int)
  parser.add_argument('--is_CP')
  return parser.parse_args()
args = parse_args(); print(args)


data_name = "s5"
num_time = 100
dim_d = args.dim_d
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

def gen_ar_beta_data(num_points, dimensions, alpha=5.0, beta=1.0, phi=0.9):

    data = np.zeros((num_points, dimensions))
    data[0] = np.random.beta(alpha, beta, dimensions)
    
    for t in range(1, num_points):
        noise = np.random.beta(5.0, 1.0, dimensions)
        data[t] = phi * data[t-1] + noise
    
    return data



if args.is_CP:
  
    num_seq = 50
    CP_time = 50

    for idx in range(num_seq):
        data = torch.zeros(num_time, dim_d)
        
        before = gen_ar_uniform_data(CP_time, dim_d)
        data[:CP_time] = torch.tensor(before, dtype=torch.float32)
        
        after = gen_ar_beta_data(num_time - CP_time, dim_d)
        data[CP_time:] = torch.tensor(after, dtype=torch.float32)

        y_list.append(data)

else:
    num_seq = 100

    for idx in range(num_seq):
    
        before = gen_ar_uniform_data(num_time, dim_d)
        data = torch.tensor(before, dtype=torch.float32)
        
        y_list.append(data)


y_list = np.array(y_list) 
print("[INFO] before:",np.mean(y_list[0][:50, :], axis=0))
print("[INFO] after:",np.mean(y_list[0][50:, :], axis=0))
print("[INFO] y_list.shape:",y_list.shape)


# Save to an .npz file
if args.is_CP:
    y_list_tensor = torch.tensor(y_list, dtype=torch.float32)
    np.savez('data/data_{}_d{}.npz'.format(data_name, dim_d), y_list = y_list_tensor)
    print('[INFO] data saved')
else:
    y_list_tensor = torch.tensor(y_list, dtype=torch.float32)
    np.savez('data/data_{}_d{}_C.npz'.format(data_name, dim_d), y_list = y_list_tensor)
    print('[INFO] data without CP saved')












