import numpy as np 
import os
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import random

import shutup 
shutup.please()

class BiasFree(nn.Module):
    
    def __init__(self, seq_len: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bias_per_window = torch.mean(x,dim=[2]) 
        seq, sensors = bias_per_window.shape
        bias_per_window = bias_per_window[:, :, None].expand([seq, sensors, self.dim]) 
        return torch.sub(x, bias_per_window), bias_per_window

class Overlapper(nn.Module):
    
    def __init__(self, seq_len: int, stride: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = seq_len
        self.stride = stride   # Define the overlap stride

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        nrows, ncolumns = input_x.shape
        
        # Calculate the number of overlapping sequences that can be extracted
        N0 = (nrows - self.dim) // self.stride + 1

        # Create overlapping sequences from the original data
        overlapping_sequences = [input_x[i*self.stride:i*self.stride+self.dim, :].unsqueeze(0) for i in range(N0)]
        final_tensor = torch.cat(overlapping_sequences, dim=0).permute(0, 2, 1)

        return final_tensor

class RandomSampler(nn.Module):
    
    def __init__(self, samples: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples = samples 

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        nrows, ncolumns, x = input_x.shape
        
        # if there is less data than needed return everything
        if nrows < self.samples:
            return input_x

        # sample
        idx = random.sample(range(0, nrows), self.samples) 
        idx.sort()
        idx = torch.tensor(idx)
        return torch.index_select(input_x, 0, idx) 


