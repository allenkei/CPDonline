import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
torch.set_printoptions(precision=5)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dim_d', default=10, type=int)
  parser.add_argument('--num_seq', default=20)
  parser.add_argument('--output_layer', default = [64,64])
  parser.add_argument('--start_t', default=20)
  parser.add_argument('--final_t', default=100)
  parser.add_argument('--epoch', default=60)
  parser.add_argument('--lr', default=0.01)

  parser.add_argument('--verbose', default=True)
  parser.add_argument('--use_data', default='s1') 
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('-f', required=False)

  return parser.parse_args()




args = parse_args(); print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)


###################
# LOAD SAVED DATA #
###################

if args.use_data == 's1':
  print('[INFO] load data_s1.npz')
  data = np.load(args.data_dir +'data_s1.npz')
  output_dir = os.path.join(f"result/s1")

os.makedirs(output_dir, exist_ok=True)


def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)



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
    output = self.l3(output)#.relu()
    return output






'''
def multivariate_basis_function(X, centers=torch.zeros(args.dim_d), sigma=1.0):

    rbf_values = torch.exp(- (X - centers) ** 2 / (2 * sigma ** 2))  # Shape: (n, d)
    phi_j_values = torch.prod(rbf_values, dim=1)  # Shape: (n,)

    return phi_j_values
'''



def kernel(u):
  return torch.exp(-0.5 * u**2) # / torch.sqrt(torch.tensor(2 * np.pi))

def kde(X, h=1.0):

  n, d = X.shape
  densities = []

  for i in range(n):
    x_eval = X[i]
    
    squared_l2_norm = torch.sum((x_eval - X) ** 2, dim=1)  # shape: (n,)

    kernel_values = kernel(squared_l2_norm / (h ** 2))  # shape: (n,)

    density = torch.sum(kernel_values) / (n * (h ** d))
    
    densities.append(density)
  
  return torch.tensor(densities)





def detect_one_seq(args, x_data, seq_iter):

  # x_data is the full_data from 1 to T

  for current_t in range(args.start_t, args.final_t+1):

    if current_t == 70: # args.start_t+1: 
      break

    # data currently seen from 1 to current_t
    current_data = x_data[:current_t] 

    print('\n[INFO] current_data.shape', current_data.shape)

    ###########################
    # GENERATE KERNEL DENSITY #
    ###########################

    #density_tensor = kde(current_data)
    #density_tensor = multivariate_basis_function(current_data)

    #print('[INFO] densities_tensor', densities_tensor)
    #print('[INFO] densities_tensor.shape', density_tensor.shape)

    ####################
    # CUSUM STATISTICS #
    ####################

    t0 = current_t
    print('[INFO] t0 =', t0)

    cusum_statistics = []

    # Calculate CUSUM for each t
    for t in range(3, t0-3):

      data_before = current_data[:t] 
      data_after = current_data[t:]
      density_before = kde(data_before)
      density_after = kde(data_after)

      print('\n')
      print('[INFO] data_before.shape', data_before.shape)
      print('[INFO] data_after.shape', data_after.shape)
      print('[INFO] density_before.shape', density_before.shape)
      print('[INFO] density_after.shape', density_after.shape)

      ###########################
      # NEURAL NETWORK (BEFORE) #
      ###########################

      model_before = CPD_online(args)
      #model_before.apply(init_weights)
      criterion_before = nn.MSELoss()
      optimizer_before = optim.Adam(model_before.parameters(), lr=args.lr)


      for epoch in range(args.epoch):
        model_before.train()
        
        pred_density_before = model_before(data_before)
        loss_before = criterion_before(pred_density_before.squeeze(), density_before)
        optimizer_before.zero_grad()
        loss_before.backward()
        optimizer_before.step()
        
        if (epoch + 1) % 10 == 0:
          print(f"Epoch [{epoch+1}/{args.epoch}], Loss (BEFORE): {loss_before.item():.4f}")

      print('[INFO] Neural Network (BEFORE) done')
      


      ##########################
      # NEURAL NETWORK (AFTER) #
      ##########################

      model_after = CPD_online(args)
      #model_after.apply(init_weights)
      criterion_after = nn.MSELoss()
      optimizer_after = optim.Adam(model_after.parameters(), lr=args.lr)


      for epoch in range(args.epoch):
        model_after.train()
        
        pred_density_after = model_after(data_after)
        loss_after = criterion_before(pred_density_after.squeeze(), density_after)
        optimizer_after.zero_grad()
        loss_after.backward()
        optimizer_after.step()
        
        if (epoch + 1) % 10 == 0:
          print(f"Epoch [{epoch+1}/{args.epoch}], Loss (AFTER): {loss_after.item():.4f}")

      print('[INFO] Neural Network (AFTER) done')


      ####################
      # CUSUM STATISTICS #
      ####################

      model_before.eval()
      with torch.no_grad():
        pred_before = model_before(data_before).squeeze()

      model_after.eval()
      with torch.no_grad():
        pred_after = model_after(data_after).squeeze()

      print('[INFO] pred_before.shape', pred_before.shape)
      print('[INFO] pred_after.shape', pred_after.shape)

      first_sum = torch.sum(pred_before) / t
      second_sum = torch.sum(pred_after) / (t0 - t)
      cusum_t = first_sum - second_sum
      cusum_statistics.append(cusum_t.item())

    # Print out the CUSUM statistics
    #print("[INFO] CUSUM Statistics length:", len(cusum_statistics))
    #print("[INFO] CUSUM Statistics:", cusum_statistics)


    x_values = list(range(1, len(cusum_statistics) + 1))
    plt.plot(x_values, cusum_statistics, marker='o', linestyle='-', label='CUSUM Statistic')
    plt.title('CUSUM Statistic vs Time Step')
    plt.savefig( output_dir + '/cusum_seq{}t{}'.format(seq_iter,t0) + '.png' ) 
    plt.close()




######################
# parameter learning #
######################


for seq_iter in range(args.num_seq):

  if seq_iter == 1: break

  x_data = torch.tensor(data['y_list'][seq_iter], dtype=torch.float32) 

  print('[INFO] data loaded with shape:', x_data.shape)

  detect_one_seq(args, x_data, seq_iter)






