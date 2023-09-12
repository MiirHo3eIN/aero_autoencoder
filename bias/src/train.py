
import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader 

import time 
import sys
import shutup 
shutup.please()

# Custom imports
sys.path.append("../../aerolib")
import modelManagement as mm
import autoencoder as ae
import dataset
import classifiers 
from models import Model

#######################################################################################################
path_Cp_data = '../../data/cp_data/AoA_0deg_Cp'

# define training and test set based on design of experiment excel document
train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50,51,55,60,64,65,69,70,73,74,79,80,83,84,88, 89,93,98,99,102,107,108,111, 112]
valid_exp = [16,27,54,61,92,103]
#######################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initData(seq_len, batch_size):
  
    print("-"*50)
    print(f"Initilaize Datasets")
    train_x = dataset.TimeseriesTensor(path_Cp_data, train_exp, seq_len=seq_len, stride=20).to(device)
    valid_x = dataset.TimeseriesTensor(path_Cp_data, valid_exp, seq_len=seq_len, stride=20).to(device)

    print(f"Train x shape: \t {train_x.shape} with {train_x.shape[0]} Training samples and {train_x.shape[2]} sequence length")
    print(f"Valid x shape: \t {valid_x.shape} with {valid_x.shape[0]} Training samples and {valid_x.shape[2]} sequence length")

    free_bias = dataset.BiasFree(seq_len)
    train_x, _= free_bias(train_x)
    valid_x, _ = free_bias(valid_x)


    dataset_train = TensorDataset(train_x , train_x)
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)

    dataset_valid = TensorDataset(valid_x, valid_x)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    del train_x, valid_x
    
    print("\n")
    return train_loader, valid_loader


if __name__ == "__main__":
    

    md = mm.createNewModel()
    md.arch_id = "2619"

    model = Model(md.arch_id)
    model.to(device)
    summary(model, input_size=(1, 36, 800))

    # This number is fixed -> see rocket repo
    md.window_size = 800

    batch_sizes = [32, 64, 128, 256, 512, 1024]
    epochs = [10, 100, 120, 160 ,200]
    alphas = [0.3, 0.5, 0.9]

    md.alpha = alphas[2]
    md.epochs = epochs[1]
    md.batch_size = batch_sizes[2]
    md.activation = "l1+alpha*mse"


    [md.latent_channels, md.latent_seq_len] = model.getLatentDim()
    md.parameters = sum(p.numel() for p in model.parameters())


    train_x, valid_x = initData(seq_len=md.window_size, batch_size=md.batch_size)
    ae.train(md, model, train_x, valid_x)

    print("-"*50)
    print(f"Saving the model: {md.model_id}")

    torch.save(model.state_dict(), f"../models/{md.model_id}.pt")

    mm.saveModel(md)
