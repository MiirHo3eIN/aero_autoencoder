import numpy as np 
import os
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import random

import shutup 
shutup.please()

# This class is not meant to be instantiated
# Its purpose is to have the mapping from experiments and labels 
# in a centalized and accesible way
class Damage_Classes():
    # Name             [ Experiments , Label, Desctiption]
    classes  =        [[range(2, 20),   0.0, "Healthy Probes"],
                       [range(20, 39),  5.0, "Healthy Probes with Additional Mass"],
                       [range(39, 58),  1.0, "5mm Crack"],
                       [range(58, 77),  2.0, "10mm Crack"],
                       [range(77, 96),  3.0, "15mm Crack"],
                       [range(96, 114), 4.0, "20mm Crack"]]

    # experiment to label
    @staticmethod
    def ex2label(experiment):
        for d_class in Damage_Classes.classes:
            if experiment in d_class[0]:
                return d_class[1]

    # label to experiments
    @staticmethod
    def label2exlist(label):
        for d_class in Damage_Classes.classes:
            if label == d_class[1]:
                return [*d_class[0]]
                
    # chech label for validity
    @staticmethod
    def validate_label(label):
        for d_class in Damage_Classes.classes:
            if label == d_class[1]:
                return True
        return False

# This class operates on the original cp data
# Creates a dataset that returns sequenced data attached with a label
class CpDataset(Dataset):
    def __init__(self, experiments: list, seq_len: int) -> None:
       

        def line_count(file_path):
            return int(os.popen(f'wc -l {file_path}').read().split()[0])
        
        # this path should not change
        self._folder_path = "../data/AoA_0deg_Cp/" 
        self._exp = experiments
        self._seq_len = seq_len

        # Some hardcoded data
        self._stride = 10 # arbitrary value
        self._skiprows = 2500 # due to some drift at the beginnig of experiments
        del_cells = [0, 23] # faulty sensors
        cols = np.arange(0, 38)
        self.use_cols = np.delete(cols, del_cells)

        # counting all samples to determine dataset length / nr of sequences
        self._c_seq = []
        self._datasetlen = 0
        for exp in self._exp:
            lines = line_count(self._folder_path+f'/aoa_0deg_Exp_{exp:03}_aerosense.csv')
            samples = lines-1-self._skiprows
            sequences = (samples - self._seq_len) // self._stride + 1
            self._datasetlen += sequences
            self._c_seq.append(self._datasetlen)
        
        print("\n---- Summary Import ----")
        print(f"Path: {self._folder_path}")
        print(f"Experiments: {self._exp}")
        print(f"Stride: {self._stride}")
        print(f"Skiprows: {self._skiprows}")
        print(f"Sequences: {self._c_seq}")
        print(f"Length: {self._datasetlen}")
        print("------------------------\n")

    def __len__(self):
        return self._datasetlen

    def __getitem__(self, seq_idx: int):

        def get_experiment_index(index: int) -> int:
            experiment_index = 0
            for seq in self._c_seq:
                if index < seq:
                    return experiment_index
                experiment_index +=1
            raise Exception("Index out of bounds")

        exp_idx = get_experiment_index(seq_idx)
        exp = self._exp[exp_idx]
        filepath = self._folder_path+f'/aoa_0deg_Exp_{exp:03}_aerosense.csv'
        df = pd.read_csv(open(filepath,'r'),
                         delimiter=' ',
                         skiprows = self._skiprows,
                         usecols = self.use_cols)
       
        # Index calculation
        if exp_idx-1 == -1:
            index_in_exp = seq_idx
        else:
            index_in_exp = seq_idx - self._c_seq[exp_idx-1]

        start =  index_in_exp * self._stride + self._skiprows
        end =  index_in_exp * self._stride + self._seq_len + self._skiprows
        mvts = df.iloc[start:end,:]

        # reformat mvts
        mvts = mvts.transpose() 
        mvts = torch.tensor(mvts.values) 

        debug = False
        if debug:
            print("Debug: ")
            print(f"seq_idx: {seq_idx}")
            print(f"exp_idx: {exp_idx}")
            print(f"exp: {exp}")
            print(f"start: {start}")
            print(f"end: {end}")
            print(f"index in experiment: {index_in_exp}")
            print(f"MVTS shape: {mvts.shape}")
        
        return mvts, Damage_Classes.ex2label(exp)


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

class TensorLoaderPickeld():
    def __init__(self, path, experiments: list) -> None:
        self._path = path 
        self._exp = experiments
        self._datasetlen = len(self._exp)

    def __len__(self) -> int:
        return self._datasetlen
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        exp = self._exp[idx]
        filepath = self._path+f'/exp_{exp:03}.pt'
        t = torch.load(filepath) 
        return t 

class TensorLoaderCp():

    def __init__(self, path, experiments: list) -> None:
        self._path = path 
        self._exp = experiments
        self._datasetlen = len(self._exp)

        #load all experiments to tensors
        del_cells = [0, 23]
        cols = np.arange(0, 38)
        self.use_cols = np.delete(cols, del_cells)
        self._skiprows = 2500

    def __len__(self) -> int:
        return self._datasetlen
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        exp = self._exp[idx]
        filepath = self._path+f'/aoa_0deg_Exp_{exp:03}_aerosense.csv'
        df = pd.read_csv(open(filepath,'r'),
                         delimiter=' ',
                         skiprows = self._skiprows,
                         usecols = self.use_cols)
       
        tensor = torch.tensor(df.values, dtype = torch.float32)

        # returns an [m, 36] tensor
        # m = used rows in dataset
        return tensor
 

# Personal note: unify the following two classes

# looks like a class, because it creates a (specific) Tensor
def TimeseriesTensor(path, experiments: list, seq_len:int, stride=10) -> torch.Tensor:

    tensors = TensorLoaderCp(path, experiments)  
    shaper = Overlapper(seq_len, stride) 

    final_tensor = None        
    for tensor in tensors:

        t = shaper(tensor)

        if  final_tensor is None:
            final_tensor = t 
        else:
            final_tensor = torch.cat((final_tensor, t), dim = 0)
        
    return final_tensor  

# This function looks like a class, because it resembles a factory.
# It creates a Dataset from a previosly pickeld dataset.
# Since it operates only on tensors, all of them are concurently in mem.
def TimeseriesPickeldTensor(folder_path, experiments: list) -> torch.Tensor:

    tensors = TensorLoaderPickeld(folder_path, experiments)  
    
    t = None        
    for tensor in tensors:

        if t is None:
            t = tensor 
        else:
            t = torch.cat((t , tensor), dim = 0)
        
    return t  

def TimeseriesSampledTensorWithLabels(folder_path, experiments: list, samples: int) -> torch.Tensor:

    tensors = TensorLoaderPickeld(folder_path, experiments)  

    sampler = RandomSampler(samples)
    
    t = None        
    labels = []
    # print(samples)
    # print(len(experiments))
    for i in range(len(experiments)):
        # print([Damage_Classes.ex2label(experiments[i])]*samples)

        tensor = sampler(tensors[i])
        labels += [Damage_Classes.ex2label(experiments[i])]*samples

        if t is None:
            t = tensor 
        else:
            t = torch.cat((t , tensor), dim = 0)
    labels = torch.Tensor(labels)
    return t, labels 

def TimeseriesSampledCpWithLabels(folder_path, experiments: list, samples: int, seq_len: int):

    tensors = TensorLoaderCp(folder_path, experiments)  
    shaper = Overlapper(seq_len=seq_len, stride=seq_len//2) 
    sampler = RandomSampler(samples)
    
    t = None        
    labels = []
    for i in range(len(experiments)):
                   
        tensor = sampler(shaper(tensors[i]))
        labels += [Damage_Classes.ex2label(experiments[i])]*samples

        if t is None:
            t = tensor 
        else:
            t = torch.cat((t , tensor), dim = 0)

    labels = torch.Tensor(labels)
    return t, labels 

# Not used at the moment
class TimeSeriesPickeld(Dataset):
    def __init__(self, path, experiments: list) -> None:
        self._path = path
        self._exp = experiments
        self.tensors = TensorLoaderPickeld(self._path, experiments)  
        
        self._stride = 10
        self._seq_len = self.tensors[0].size(dim=2)
        # counting all samples to determine dataset length / nr of sequences
        self._c_seq = []
        self._datasetlen = 0
        for t in self.tensors:
            seq = t.size(dim=0)
            self._datasetlen += seq 
            self._c_seq.append(self._datasetlen)
        
        print("\n---- Summary Import ----")
        print(f"Path: {self._path}")
        print(f"Experiments: {self._exp}")
        print(f"Stride: {self._stride}")
        print(f"Sequences: {self._c_seq}")
        print(f"Length: {self._datasetlen}")
        print(f"Seq Length {self._seq_len}")
        print("------------------------\n")

        # print(f"Test Value: {self.tensors[0][0].shape}")
        #
        # print("------------------------\n")
    def __len__(self):
        return self._datasetlen

    def __getitem__(self, seq_idx: int):

        def get_experiment_index(index: int) -> int:
            experiment_index = 0
            for seq in self._c_seq:
                if index < seq:
                    return experiment_index
                experiment_index +=1
            raise Exception("Index out of bounds")

        exp_idx = get_experiment_index(seq_idx)
        exp = self._exp[exp_idx]
        
        # Index calculation
        if exp_idx-1 == -1:
            index_in_exp = seq_idx
        else:
            index_in_exp = seq_idx - self._c_seq[exp_idx-1]

        mvts = self.tensors[exp_idx][index_in_exp]

        debug = False 
        if debug:
            print("Debug: ")
            print(f"seq_idx: {seq_idx}")
            print(f"exp_idx: {exp_idx}")
            print(f"exp: {exp}")
            print(f"index in experiment: {index_in_exp}")
            print(f"MVTS shape: {mvts.shape}")
        
        return mvts, Damage_Classes.ex2label(exp)

if __name__ == "__main__":
    # timeseries = CpDataset([2,3], 200)
    # ts, label = timeseries[455]
    # print(ts.shape)

    # path = "../data/pickeld/CA5B:E21B:71ED:3A1C-2023_08_15_14_41_11/reconstructed"
    # TimeseriesPickeldTensor(path, [3])

    # t = TimeseriesTensor("../data/AoA_0deg_Cp/", [3], 200)
    # print(t.shape)
    seq_len = 2000
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]
    path_data = '../data/cp_data_true/AoA_0deg_Cp'
    train_x, train_labels = TimeseriesSampledCpWithLabels(path_data, test_exp, 10, seq_len)
    # x, labels = TimeseriesSampledTensorWithLabels(path_path, train_exp, 10, cp=True, seq_len=seq_len)
    print(train_x.shape)
    pass
