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
  parser.add_argument('--num_seq', default=50)
  parser.add_argument('--start_t', default=30)
  parser.add_argument('--final_t', default=100)
  parser.add_argument('--t_gap', default=3)

  parser.add_argument('--output_layer', default = [64,64])
  parser.add_argument('--epoch', default=40)
  parser.add_argument('--lr', default=0.001)
  #parser.add_argument('--L2_samples', default=10000)
  parser.add_argument('--C2', type=float)
  parser.add_argument('--alpha', default=0.5)

  parser.add_argument('--verbose', default=False)
  parser.add_argument('--save_fig', default=False)
  parser.add_argument('--use_data', default='s1')
  parser.add_argument('--quantile')
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


print('[INFO] load data_{}_d{}.npz'.format(args.use_data, args.dim_d))
data = np.load(args.data_dir + 'data_{}_d{}.npz'.format(args.use_data, args.dim_d))
output_dir = os.path.join(f"result/{args.use_data}")


os.makedirs(output_dir, exist_ok=True)


def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)


# global variable
C2 = args.C2; alpha = args.alpha; D = args.dim_d 


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




def detect_one_seq(args, x_data, seq_iter):
    # x_data is the full_data from 1 to T
    
    for t0 in range(args.start_t, args.final_t):

        
        current_data = x_data[:t0] # data currently seen from 1 to t0

        ratio_holder = [0] * t0 # store l2_norm / deno_G

        
        # half of time span
        n = math.ceil(math.log2(t0 / 7)) # 7 is a threshold
        binary_cut = []
        current_value = t0-args.t_gap
        binary_cut.append(current_value)
        for i in range(n):
            current_value /= 2
            binary_cut.append(int(math.ceil(current_value)))


        #list_of_t = list(range(args.t_gap, t0 - args.t_gap + 1, args.t_gap))
        #print("[INFO] t0 =", t0, ", list_of_t:", list_of_t)


        #### OPTIONS:
        #### for t in range(args.t_gap, t0-args.t_gap):
        #### for t in list_of_t:
        #### for t in binary_cut: 

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


            deno_G = torch.tensor( (t0 - t)**(-alpha / (2 * alpha + D)) )
            ratio = func_l2_norm / deno_G
            ratio_holder[t] = ratio.item()
            # END LOOP OVER t



        if args.save_fig: 
            tau_values = [C2] * t0 # ratio compare with C2
            x_values = list(range(1, len(ratio_holder) + 1))
            plt.plot(x_values, ratio_holder, marker='o', linestyle='-', label='CUSUM Statistic')
            plt.plot(x_values, tau_values, color='red', label='Threshold tau_t', linestyle='--')
            plt.title('ratio vs Time Step seq{} t{}'.format(seq_iter,t0))
            plt.savefig( output_dir + '/cusum_seq{}t{}'.format(seq_iter,t0) + '.png' ) 
            plt.close()

        #############
        # DETECTION #
        #############
        # raise an alarm at t0
        if max(ratio_holder) > C2:
            print("[INFO] Alarm raised at current_t:", t0)
            break
        # END LOOP OVER t0


    if t0 == (args.final_t-1):
      print("[INFO] No CP detected!")
      
    return(t0)



##############
# EVALUATION #
##############

def eval(output_t0, Delta, T):
    
    N = len(output_t0)
    delay_sum = torch.sum((output_t0[output_t0 >= Delta] - Delta).float()) # numerator
    delay_count = torch.sum(output_t0 >= Delta).float() # denominator

    pfa_count = torch.sum(output_t0 < Delta).float() # numerator
    pfn_count = torch.sum(output_t0 == T).float() # numerator

    delay = delay_sum / delay_count
    pfa = pfa_count / N
    pfn = pfn_count / N

    result = torch.stack([delay, pfa, pfn])
    return result

####################
# ONLINE DETECTION #
####################


output_t0 = torch.zeros(args.num_seq)


for seq_iter in range(0,args.num_seq):

  print("[INFO] seq_iter =", seq_iter)

  x_data = data['y_list'][seq_iter] # numpy
  output_t0[seq_iter] = detect_one_seq(args, x_data, seq_iter)
  holder_df = pd.DataFrame(output_t0.cpu().numpy())
  file_path = output_dir + '/result_{}_d{}_q{}.csv'.format(args.use_data, args.dim_d, args.quantile)
  holder_df.to_csv(file_path, mode='w', header=True, index=False)



result = eval(output_t0, Delta=50, T=args.final_t-1)
print('Eval Metrics:', result)
result_df = pd.DataFrame(result.cpu().numpy())
file_path = output_dir + '/metric_{}_d{}_q{}.csv'.format(args.use_data, args.dim_d, args.quantile)
result_df.to_csv(file_path, mode='w', header=True, index=False)






