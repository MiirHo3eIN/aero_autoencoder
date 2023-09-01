import torch
import numpy as np 
import matplotlib.pyplot as plt 

input_x = torch.randn(10000, 36)


seq_len = 200

nrows, ncolumns = input_x.shape
dim =  seq_len
stride = 1  # Define the overlap stride

# Calculate the number of overlapping sequences that can be extracted
N0 = (nrows - dim) // stride + 1

# Create overlapping sequences from the original data
overlapping_sequences = [input_x[i*stride:i*stride+dim, :].unsqueeze(0) for i in range(N0)]
final_tensor = torch.cat(overlapping_sequences, dim=0)

final_tensor_permuted = final_tensor.permute(0, 2, 1)

print(final_tensor.shape)

plt.figure()

plt.plot(input_x[:200, 0].detach().numpy(), color = 'green', label = 'original')
plt.plot(final_tensor[0, :, 0].detach().numpy(), color = 'red', label = 'shaped signal')
plt.plot(final_tensor_permuted[0, 0,:].detach().numpy(), color = 'black', label = 'shaped signal')
plt.legend()

plt.show()