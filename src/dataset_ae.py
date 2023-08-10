import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import signal

import shutup 

shutup.please()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class MeanDataset(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.mean(x, dim=0)
    
class HighPassFilter(nn.Module): 

    def __init__(self, sampling_frequency, cutoff_frequency, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._sampling_frequency = sampling_frequency
        self._cutoff_frequency = cutoff_frequency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Going from torch to numpy 
        x = x.detach().cpu().numpy()

        x = self.__high_pass_filter__(x, 
                                fs = self._sampling_frequency, 
                                cutoff_freq = self._cutoff_frequency,
                                order = 4)
        
        
        return torch.from_numpy(x).float() 

    def __high_pass_filter__(self, data: np.array, fs = 100, cutoff_freq = 0.5, order = 4): 
    
        # Design the filter
        b, a = signal.butter(order, cutoff_freq, fs=fs, btype='high', analog=False, output='ba')
        # Apply the high-pass filter to the input signal
        filtered_signal = signal.lfilter(b, a, data)
        return filtered_signal


class data_shape(nn.Module):
    
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



class RawDataset: 

    def __init__(self, folder_path, experiments: np.array, seq_len: int) -> None:

        self._folder_path = folder_path 
        self._eperiments = experiments
        self._datasetlen = len(experiments)
        self._seq_len = seq_len 


        self._mean = MeanDataset()
        self._high_pass_filter = HighPassFilter(sampling_frequency= 100, cutoff_frequency= 0.5)
        self.__shape_data = data_shape(seq_len = self._seq_len, stride = 10)

    def __len__(self) -> int:
        return self._datasetlen
    

    def __getitem__(self, idx: int) -> np.array:

         
        del_cells = [0, 23]
        cols = np.arange(0, 38)
        use_cols_ = np.delete(cols, del_cells)
        columns_ = [f"sensor_{sense}" for sense in use_cols_]

        labels = torch.tensor([])

        experiment = self._eperiments[idx]
        exp_num_three_digit = str(experiment).zfill(3)
        filename = '/aoa_0deg_Exp_'+str(exp_num_three_digit)+'_aerosense.csv'
        
        filepath = self._folder_path+filename
        df = pd.read_csv(open(filepath,'r'), delimiter=' ', skiprows = 2500, usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,31,32,33,34,35,36,37,38])
        df.columns = columns_
        torch_df = torch.tensor(df.values, dtype = torch.float32)
        
        final_tensor = self.__shape_data(torch_df)
        #final_tensor = torch.flatten(final_tensor, start_dim=0, end_dim = 1)

        return final_tensor 




def TimeSeriesDataset(folder_path, experiments: np.array, seq_len:int) -> torch.Tensor:

    dataset = RawDataset(folder_path, experiments, seq_len= seq_len)  
    
    sample_acum = 0
    for tensor_num in np.arange(0, len(dataset)):
        
        tensor = dataset[tensor_num]

        if tensor_num == 0:
            tensors_cat = tensor
        else:
            tensors_cat = torch.cat((tensors_cat, tensor), dim = 0)
        
        sample_acum += tensor.shape[0]

#    assert tensors_cat.shape[0] ==  sample_acum , "Concatenation of tensors is not working" 
    
    return tensors_cat  


def func_test():

    folder_path = "/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_0deg_Cp"
    train_experiments = np.arange(5, 10)
    seq_len = 100
    train_x = TimeSeriesDataset(folder_path, train_experiments, seq_len = seq_len)


#if __name__ == "__main__":
#    func_test()