import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import random
import os
import math
import pandas as pd
torch.set_printoptions(precision=5)



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dim_d', default=10, type=int)
  parser.add_argument('--num_seq', default=100)
  parser.add_argument('--start_t', default=30)
  parser.add_argument('--final_t', default=100)
  parser.add_argument('--t_gap', default=3)

  parser.add_argument('--output_layer', default = [64,64])
  parser.add_argument('--epoch', default=40)
  parser.add_argument('--lr', default=0.001)
  #parser.add_argument('--L2_samples', default=10000)
  parser.add_argument('--alpha', default=0.5)

  parser.add_argument('--verbose', default=False)
  parser.add_argument('--use_data', default='s1') 
  parser.add_argument('--data_dir', default='./data/')

  return parser.parse_args()


args = parse_args()
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)
print(args)


###################
# LOAD SAVED DATA #
###################


print('[INFO] load data_{}_d{}_C.npz'.format(args.use_data, args.dim_d))
data = np.load(args.data_dir + 'data_{}_d{}_C.npz'.format(args.use_data, args.dim_d))


def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)


# global variable
alpha = args.alpha; D = args.dim_d 


#########
# MODEL #
#########


class CPD_online(nn.Module):
  def __init__(self, args):
    super(CPD_online, self).__init__()

    self.d = args.dim_d
    self.l1 = nn.Linear( self.d, args.output_layer[0])
    self.l2 = nn.Linear( args.output_layer[0], args.output_layer[1])
    self.l3 = nn.Linear( args.output_layer[1], 1)

  def forward(self, x):
    output = self.l1(x).relu()
    output = self.l2(output).relu()
    output = self.l3(output)
    return output




def calculate_C_one_seq(args, x_data, seq_iter):
    # x_data is the full_data from 1 to T

    # storing C for this sequence
    C_holder_one_seq = torch.zeros(args.final_t) # length of T
    
    for t0 in range(args.start_t, args.final_t):
    
        current_data = x_data[:t0] # data currently seen from 1 to t0

        ratio_holder = torch.zeros(t0) # store l2_norm / deno_G
        
        # Choice of t
        n = math.ceil(math.log2(t0 / 7)) # 7 is a threshold
        binary_cut = []
        current_value = t0-args.t_gap
        binary_cut.append(current_value)
        for i in range(n):
            current_value /= 2
            binary_cut.append(int(math.ceil(current_value)))


        for t in binary_cut:

            criterion = nn.MSELoss(reduction="sum")

            # BEFORE
            data_before = current_data[:t] # X numpy
            kde_before = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_before)
            log_density_before = kde_before.score_samples(data_before)
            density_before = torch.tensor(np.exp(log_density_before), dtype=torch.float32).to(device) # Y torch tensor
            data_before = torch.tensor(data_before, dtype=torch.float32).to(device) # X torch tensor

            # AFTER
            data_after = current_data[t:] # X numpy
            kde_after = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_after)
            log_density_after = kde_after.score_samples(data_after)
            density_after = torch.tensor(np.exp(log_density_after), dtype=torch.float32).to(device) # Y torch tensor
            data_after = torch.tensor(data_after, dtype=torch.float32).to(device) # X torch tensor


            ###########################
            # NEURAL NETWORK (BEFORE) #
            ###########################

            model_before = CPD_online(args).to(device)
            model_before.apply(init_weights)
            optimizer_before = optim.Adam(model_before.parameters(), lr=args.lr)

            for epoch in range(args.epoch):
              model_before.train()
              pred_density_before = model_before(data_before)
              loss_before = criterion(pred_density_before.squeeze(), density_before)
              optimizer_before.zero_grad()
              loss_before.backward()
              optimizer_before.step()
              
              
              if args.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epoch}], Loss (BEFORE): {loss_before.item():.4f}")
              

            ##########################
            # NEURAL NETWORK (AFTER) #
            ##########################

            model_after = CPD_online(args).to(device)
            model_after.apply(init_weights)
            optimizer_after = optim.Adam(model_after.parameters(), lr=args.lr)

            for epoch in range(args.epoch):
              model_after.train()
              
              pred_density_after = model_after(data_after)
              loss_after = criterion(pred_density_after.squeeze(), density_after)
              optimizer_after.zero_grad()
              loss_after.backward()
              optimizer_after.step()

              
              if args.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epoch}], Loss (AFTER): {loss_after.item():.4f}")
              

            ####################
            # CUSUM STATISTICS #
            ####################

            #sampled_x = ( 2*torch.rand(args.L2_samples, args.dim_d).to(device)-1 ) * 5
            sampled_x = torch.from_numpy(current_data.copy())

            model_before.eval()
            with torch.no_grad():
              pred_before = model_before(sampled_x).squeeze() # use sampled_x

            model_after.eval()
            with torch.no_grad():
              pred_after = model_after(sampled_x).squeeze() # use sampled_x

            first_term = pred_before * torch.sqrt( torch.tensor( (t0-t) / (t*t0) ) )# * t
            second_term = pred_after * torch.sqrt( torch.tensor( (t) / ((t0-t)*t0) ) )# * (t0-t)
            squared_outputs = (first_term - second_term) ** 2
            func_l2_norm = torch.sqrt(torch.mean(squared_outputs))

            deno_G = torch.tensor( (t0 - t)**(-alpha / (2 * alpha + D)) ) # vary by t
            ratio = func_l2_norm / deno_G
            ratio_holder[t] = ratio
            # END LOOP OVER t 

        # within the loop of t0
        C_holder_one_seq[t0] = max(ratio_holder) # max over t
        # END LOOP OVER t0

    return C_holder_one_seq


    

###############
# CALCULATE C #
###############


holder_to_save = torch.zeros(args.num_seq, args.final_t)


for seq_iter in range(0,args.num_seq):

    if (seq_iter+1) % 20 == 0: 
      max_holder, _ = torch.max(holder_to_save, axis=1)
      print("[INFO] seq_iter =", seq_iter+1, "| quantile =", np.percentile(max_holder.cpu().numpy(), 95))
    
    x_data = data['y_list'][seq_iter] # numpy
    holder_to_save[seq_iter,:] = calculate_C_one_seq(args, x_data, seq_iter)
    holder_df = pd.DataFrame(holder_to_save.cpu().numpy())
    file_path = 'data/C_{}_d{}.csv'.format(args.use_data, args.dim_d)
    holder_df.to_csv(file_path, mode='w', header=True, index=False)
    #print(f"Saved updated file at iteration {seq_iter+1}")






