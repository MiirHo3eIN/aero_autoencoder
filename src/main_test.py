import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary
import os
import math
import pandas as pd


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


def criterion(x, x_hat):
    mse = nn.MSELoss()
    return mse(x.float(), x_hat.float())


def model_eval(model_id, tiny=False):
    
    # Hardcoded Data
    path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
    path_results = "../training_results.csv"
    path_models = f'../trained_models/{model_id}.pt'
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

    # Load the Testdata
    df = pd.read_csv(open(path_results))
    seq_len = df[df["model_id"] == model_id]['window_size'].values[0]
    test_x = TimeSeriesDataset(path_Cp_data, test_exp, seq_len = seq_len)    
    
    # Load the Model
    model = CNN_AE(c_in = 36, tiny=tiny)
    model.load_state_dict(torch.load(path_models))    

    # Run the Testdata through the Model
    with torch.no_grad():
        model.eval() 
        test_x_hat = model(test_x.float())
    
    # Calculate the metric
    output = criterion(test_x.float(), test_x_hat.float())

    # Print the Results into a Plot
    sensor = 17
    title = f'Model: {model_id}  | Seq_length: {seq_len} \n MSE: {output} \n Sensor: {sensor}'
    with sns.plotting_context("poster"):
        sns.set(rc={'figure.figsize':(15,8.27)})
        plt.figure()
        plt.plot(test_x[0, sensor-1, :].detach().numpy(), color = 'green', label = 'original')
        plt.plot(test_x_hat[0, sensor-1, :], color = 'red', label = 'reconstructed signal')
        plt.legend()
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Cp")
        plt.savefig(f"../plots/tests/test_{model_id}.png")
        plt.show()




if __name__ == "__main__":
    cnn = [ "CA5B:E21B:71ED:3A1C", "F06D:D524:BFD6:232E", "D86A:2185:C32B:7239", "A3B3:8C1F:43AC:7718", "B4AD:31CC:3620:B782"] 
    tiny_cnn = ["7547:B8DA:C870:507A", "829C:AF16:5D58:E61C", "C019:A640:74EF:D675", "102E:5B5E:C956:FD77"]

    model_eval(tiny_cnn[3], tiny=True)
#   for model in cnn:
#      model_eval(model)
 
#    for model in tiny_cnn:
#        model_eval(model, tiny=True)
