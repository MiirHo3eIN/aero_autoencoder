import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset

import tqdm
from tqdm.notebook import tqdm_notebook

import random

import shutup 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
shutup.please()

# Custom imports
from dataset_ae import TimeSeriesDataset
from ae_model import linear_autoencoder, CNN_AE


# User input

path_Cp_data = '/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_0deg_Cp'
path_saved_features =  '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_0deg/'


test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

seq_len = 200

def plot_reconstructed_signal(test_x, test_x_hat, sensors):
        
        with sns.plotting_context("poster"):
            for idx in sensors:
                plt.figure()
                plt.plot(test_x[0, idx-1, :].detach().numpy(), color = 'green', label = 'original')
                plt.plot(test_x_hat[0, idx-1, :], color = 'red', label = 'reconstructed signal')
                plt.legend()
                plt.title(f"Sensor {idx}")
                plt.xlabel("Samples")
                plt.ylabel("Cp")
        plt.show()


def main_eval(): 

    test_x = TimeSeriesDataset(path_Cp_data, test_exp, seq_len = seq_len)    

    model_id = "CA5B:E21B:71ED:3A1C.pt" # 
    model = CNN_AE(c_in = 36)
    models_path = f'../trained_models/{model_id}'
    model.load_state_dict(torch.load(models_path))    
    model.eval()
    
    test_x = (test_x.float())

    
    test_x_hat = model(test_x).detach().numpy()

    print(test_x.shape)
    print(test_x_hat.shape)
    cluster_1 = [1,2,3,4,5,6,7,8]
    cluster_2 = [9,10,11,12,13,14,15,16]
    plot_reconstructed_signal(test_x, test_x_hat, sensors = [1])




if __name__ == "__main__":
    main_eval()