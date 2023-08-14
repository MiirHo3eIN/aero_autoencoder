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
from datetime import datetime
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
from ae_model import Models
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
from ae_model import Models





def main(model_id):
    # Hardcoded Data
    path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
    path_results = "../training_results.csv"
    path_models = f'../trained_models/{model_id}.pt'

    path_dataset_parent =  '../../data/compressed_datasets/'

    path_dataset_child = f"{model_id}-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    path_dataset = path_dataset_parent+"/"+path_dataset_child+"/"

    dir_ = os.mkdir(path_dataset)

    train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50,51,55,60,64,65,69,70,73,74,79,80,83,84,88, 89,93,98,99,102,107,108,111, 112]
    valid_exp = [16,27,54,61,92,103]
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]
 
    experiments = train_exp + valid_exp + test_exp
    # Load the Testdata
    df = pd.read_csv(open(path_results))
    seq_len = df[df["model_id"] == model_id]['window_size'].values[0]
    latent = df[df["model_id"] == model_id]['latent_dim'].values[0]
    
    # Load the Model
    model = Models.get(model_id)
    model.load_state_dict(torch.load(path_models))    

    for experiment in experiments:
        
        # Load the data 
        test_x = TimeSeriesDataset(path_Cp_data, [experiment], seq_len = seq_len)    

        # Run the data through the model
        with torch.no_grad():
            model.eval() 
            test_x_hat = model(test_x.float())
    
        # pickle reconstructed signal along with original signal
        filename_original = f"exp_{experiment}_original.pt"
        filename_reconstructed = f"exp_{experiment}_reconstructed.pt"
        torch.save(test_x, path_dataset + filename_original)
        torch.save(test_x_hat, path_dataset + filename_reconstructed)


    meta = dict()
    



if __name__ == "__main__":
    cnn = [ "CA5B:E21B:71ED:3A1C", "F06D:D524:BFD6:232E", "D86A:2185:C32B:7239", "A3B3:8C1F:43AC:7718", "B4AD:31CC:3620:B782"] 
    tiny_cnn = ["7547:B8DA:C870:507A", "829C:AF16:5D58:E61C", "C019:A640:74EF:D675", "102E:5B5E:C956:FD77"]

    # model_eval(tiny_cnn[3])
    main(cnn[4])
    print("DOne")
    # for model in cnn:
    #     model_eval(model)
 
    # for model in tiny_cnn:
    #     model_eval(model)

