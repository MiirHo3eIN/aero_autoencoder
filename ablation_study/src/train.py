
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




def initData(seq_len, batch_size):
  
    print("-"*50)
    print(f"Initilaize Datasets")
    train_x = dataset.TimeseriesTensor(path_Cp_data, train_exp, seq_len=seq_len, stride=20).to(device)
    valid_x = dataset.TimeseriesTensor(path_Cp_data, valid_exp, seq_len=seq_len, stride=20).to(device)

    print(f"Train x shape: \t {train_x.shape} with {train_x.shape[0]} Training samples and {train_x.shape[2]} sequence length")
    print(f"Valid x shape: \t {valid_x.shape} with {valid_x.shape[0]} Training samples and {valid_x.shape[2]} sequence length")

    dataset_train = TensorDataset(train_x , train_x)
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)

    dataset_valid = TensorDataset(valid_x, valid_x)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)
    del train_x, valid_x
    
    print("\n")
    return train_loader, valid_loader




def train(model, train_x, valid_x, epochs, alpha):

    print("-"*50)
    print("Setup Training...")
    lr = 0.0001
    betas = (0.9, 0.98)
    eps = 1e-9
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    criterion = reconstruction_loss(alpha=alpha)
    print(f"Used Device: {device}")
    print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
    print(f"Criterion alpha: {alpha}")
    print("\n")


    train_epoch_loss, valid_epoch_loss = [], []
    train_batch_loss, valid_batch_loss = [], []
    train_total_loss, valid_total_loss = [], []
    
    print("-"*50)
    print("Starting Training...")
    time_start = time.time()
    for epoch in np.arange(0, epochs):
        
        print(f"Epoch: {epoch+1}/{epochs}", end="")

        ### TRAINING PHASE ###
        for x_batch, y_batch in (train_x):
            # print("x_batch shape: ", x_batch.shape)
            # print("y_batch shape: ", y_batch.float().shape)
            
           
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
        for x_batch, y_batch in (valid_x):

            with torch.no_grad():
                model.eval()

                y_valid = model.forward(x_batch.float())
                valid_loss = criterion(y_valid.float(), y_batch.float())

                valid_batch_loss += [valid_loss]
                valid_epoch_loss += [valid_loss.item()]


        print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
                Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")

        train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
    time_end = time.time()
    train_time = time_end - time_start

    return [train_time, train_total_loss[-1], valid_total_loss[-1]] 




if __name__ == "__main__":
    
    # This number is fixed -> see rocket repo
    seq_len = 800

    batch_sizes = [32, 64, 128, 256, 512, 1024]
    epochs = [10, 100, 120, 160 ,200]
    alphas = [0.3, 0.5, 0.9]


    md = mm.createNewModel()
    md.arch_id = "068f"
    md.alpha = alphas[2]
    md.epoch = epochs[0]
    md.batch_size = batch_sizes[0]
    md.activation = "l1+alpha*mse"


    # This number is fixed -> see rocket repo
    md.window_size = 800

    model = Model(md.arch_id)
    model.to(device)

    [md.latent_channels, md.latent_seq_len] = model.getLatentDim()
    md.parameters = sum(p.numel() for p in model.parameters())

    summary(model, input_size=(1, 36, seq_len))

    train_x, valid_x = initData(seq_len=md.window_size, batch_size=md.batch_size)
    results = train(model, train_x, valid_x, md.epoch, md.alpha)

    print("-"*50)
    print(f"Saving the model: {md.model_id}")

    torch.save(model.state_dict(), f"../models/{md.model_id}.pt")

    md.train_loss = results[2]
    md.valid_loss = results[1] 
    md.train_time = results[0] 

    mm.saveModel(md)
