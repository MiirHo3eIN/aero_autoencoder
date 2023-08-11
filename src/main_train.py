import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary


import plotly.graph_objects as go
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, TensorDataset

import time 

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
import data_save 

# User input

path_Cp_data = '/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_0deg_Cp'
path_saved_features =  '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_0deg/'

# define training and test set based on design of experiment excel document

train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50,51,55,60,64,65,69,70,73,74,79,80,83,84,88, 89,93,98,99,102,107,108,111, 112]
valid_exp = [16,27,54,61,92,103]



data_single_node = {
    "model_id": None,
    "architecture": None,
    "activation": None,
    "window_size": None,
    "latent_dim": None,
    "train_loss": None,
    "valid_loss": None,
    "train_time": None
}

class reconstruction_loss(nn.Module):
    def __init__(self, alpha, *args, **kwargs) -> None:    
        super().__init__(*args, **kwargs)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_loss = self.criterion1(x, y)
        l1_loss = self.criterion2(x, y)
        return (l1_loss + self.alpha*mse_loss) / (self.alpha)


def main_train(seq_len, batch_size, epochs, alpha):
    
    print("Enter main")

    # Hyper parameters

    # Lin = Input Vector length 
    # Lout = Output Vector length
    # batch_size  
    # epochs 


    train_x = TimeSeriesDataset(path_Cp_data,train_exp, seq_len= seq_len)
    valid_x = TimeSeriesDataset(path_Cp_data,valid_exp, seq_len= seq_len)
    print(train_x.shape)

    
    print("-"*50)
    
    print(f"Train x shape: \t {train_x.shape} with {train_x.shape[0]} Training samples and {train_x.shape[1]} sequence length")
    print(f"Valid x shape: \t {valid_x.shape} with {valid_x.shape[0]} Training samples and {valid_x.shape[1]} sequence length")

    model = CNN_AE(c_in=36 )

    summary(model, input_size=(1, 36, seq_len))

    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = reconstruction_loss(alpha= alpha)

    dataset_train = TensorDataset(train_x , train_x)
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)

    dataset_valid = TensorDataset(valid_x, valid_x)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    del train_x, valid_x

    train_epoch_loss, valid_epoch_loss = [], []
    train_batch_loss, valid_batch_loss = [], []
    train_total_loss, valid_total_loss = [], []
    model_number = data_save.generate_hexadecimal()
    
    time_start = time.time()
    for epoch in np.arange(0, epochs):
        
        print(f"Epoch: {epoch+1}/{epochs}")

        ### TRAINING PHASE ###

        for x_batch, y_batch in (train_loader):
            #print("x_batch shape: ", x_batch.shape)
            #print("y_batch shape: ", y_batch.float().shape)
            
           
            y_train = model.forward(x_batch.float())
            train_loss = criterion(y_train.float(), y_batch.float())


            train_batch_loss += [train_loss]
            train_epoch_loss += [train_loss.item()]

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()

            # Update the parameters
            optimizer.step()

              
        ### VALIDATION PHASE ###
        
        for x_batch, y_batch in (valid_loader):

            with torch.no_grad():
                model.eval()

                y_valid = model.forward(x_batch.float())
                valid_loss = criterion(y_valid.float(), y_batch.float())

                valid_batch_loss += [valid_loss]
                valid_epoch_loss += [valid_loss.item()]

        # Save the model 
        torch.save(model.state_dict(), f"../trained_models/{model_number}.pt")

        print(f"epoch{epoch+1}, \t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss)}, \
        Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss)}")

        train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
    time_end = time.time()
    train_time = time_end - time_start

    print("Saving the model information")
    data_single_node["model_id"] = model_number
    data_single_node["architecture"] = "CNN-Based"
    data_single_node["activation"] = "l1+0.9mse"
    data_single_node["window_size"] = seq_len
    data_single_node["latent_dim"] = seq_len//8
    data_single_node["train_loss"] = train_total_loss[-1]
    data_single_node["valid_loss"] = valid_total_loss[-1]
    data_single_node["train_time"] = train_time


    data_save.write_to_csv(data_single_node)


    # Plot the loss
    #plt.figure(figsize=(12, 8))
    #plt.plot(train_total_loss, label='Train loss')
    #plt.plot(valid_total_loss, label='Valid loss')
    #plt.legend()
    #plt.show()
   

if __name__ == "__main__":
    
    seq_len_list = [200, 400, 600, 800 ]
    batch_size = [32, 64, 128, 256, 512, 1024]
    epochs = [100, 120, 160 ,200]
    alpha = [0.3, 0.5, 0.9]

    for seq_len in seq_len_list:
        print(f"Training for Sequence length: {seq_len}")
        main_train(seq_len, batch_size[2], epochs[-2], alpha[-1])
        print("Done")