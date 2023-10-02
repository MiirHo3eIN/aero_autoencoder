import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader 

import time 

import shutup 
shutup.please()


import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from dataset import * 
from ae_model import *
import utils 

# User input
torch.device("cpu")
path_Cp_data = '../data/AoA_0deg_Cp'

# define training and test set based on design of experiment excel document
train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50] #,51,55,60,64,65,69,70,73,74,79,80,83,84,88] #, 89,93,98,99,102,107,108,111, 112]
valid_exp = [16,27,54,61,92,103]

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_single_node = {
    "model_id": None,
    "arch_id": None,
    "activation": None,
    "window_size": None,
    "latent_channels": None,
    "latent_seq_len": None,
    "train_loss": None,
    "valid_loss": None,
    "train_time": None,
    "mse": None
}


def plot(x_input, color, label ): 
    with sns.plotting_context("poster"):
        plt.figure(figsize = (16,3))
        plt.plot(x_input, color = color, label = label)
        plt.legend()


class reconstruction_loss(nn.Module):
    def __init__(self, alpha, *args, **kwargs) -> None:    
        super().__init__(*args, **kwargs)
        self.criterion1 = nn.MSELoss(reduction='mean')
        self.criterion2 = nn.L1Loss(reduction='mean')
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_loss = self.criterion1(x, y)
        l1_loss = self.criterion2(x, y)
        return (l1_loss + self.alpha*mse_loss) / (self.alpha)

      
def initData(seq_len, stride, batch_size):
  
    print("-"*50)
    print(f"Initilaize Datasets")
    train_x = TimeseriesTensor(path_Cp_data,train_exp, seq_len= seq_len)
    valid_x = TimeseriesTensor(path_Cp_data,valid_exp, seq_len= seq_len)
    train_x, valid_x = train_x , valid_x 

    print(f"Train x shape: \t {train_x.shape} with {train_x.shape[0]} Training samples and {train_x.shape[2]} sequence length")
    print(f"Valid x shape: \t {valid_x.shape} with {valid_x.shape[0]} Training samples and {valid_x.shape[2]} sequence length")

    dataset_train = TensorDataset(train_x , train_x)
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False )

    dataset_valid = TensorDataset(valid_x, valid_x)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False )
    

    
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
    #print(f"Used Device: {device}")
    print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
    print(f"Criterion alpha: {alpha}")
    print("\n")


    train_epoch_loss, valid_epoch_loss = [], []
    train_batch_loss, valid_batch_loss = [], []
    train_total_loss, valid_total_loss = [], []
    
    print("-"*50)
    print("Starting Training...")
    time_start = time.time()
    print(time_start)

    
    
    for epoch in np.arange(0, epochs):
        #print("Inside the for loop")
        print(f"Epoch: {epoch+1}/{epochs}")

        ### TRAINING PHASE ###

        for x_batch, y_batch in (train_x):
            # print("x_batch shape: ", x_batch.shape)
            # print("y_batch shape: ", y_batch.float().shape)
            
           
            y_train = model.forward(x_batch.float())
            train_loss = criterion(y_train.float(), y_batch.float())

            #train_loss = torch.min(train_loss)
            train_batch_loss += [train_loss]
            train_epoch_loss += [train_loss.item()]
            #train_epoch_loss += [torch.mean(item) for item in  (train_loss.item())]

            # Backpropagation
#            train_loss = train_loss.sum()
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
                valid_epoch_loss += [(valid_loss.item())]


        print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
                Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")

        train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
    time_end = time.time()
    train_time = time_end - time_start

    return train_time , train_total_loss, valid_total_loss 

def save_model(model, archID, seq_len, train_loss, valid_loss, train_time, interactive=True):
    l_channels = "unknown"
    l_seq_len = "unknown"
    if interactive:
        answer = input("Do you want to save the Model? (y/n): ")
        if answer == "n": exit()

        answer = input("Please enter the latent channels? [0-100]: ")
        l_channels = utils.checkNumber(answer)
    
        answer = input("Please enter the latent sequence length? [0-10000]: ")
        l_seq_len = utils.checkNumber(answer)

    model_number = utils.generate_hexadecimal()

    print("-"*50)
    print(f"Saving the model: {model_number}")

    torch.save(model.state_dict(), f"../trained_models/{model_number}.pt")

    data_single_node["model_id"] = model_number
    data_single_node["arch_id"] = archID
    data_single_node["activation"] = "l1+0.9mse"
    data_single_node["window_size"] = seq_len
    data_single_node["latent_seq_len"] = l_seq_len
    data_single_node["latent_channels"] = l_channels
    data_single_node["train_loss"] = train_loss
    data_single_node["valid_loss"] = valid_loss 
    data_single_node["train_time"] = train_time

    utils.write_to_csv(data_single_node)


if __name__ == "__main__":
    
    seq_len_list = [200, 400, 600, 800 ]
    batch_size = [32, 64, 128, 256, 512, 1024]
    epochs = [10, 100, 120, 160 ,200]
    alphas = [0.3, 0.5, 0.9]

    archID = "a61c"
    model = Model(archID, 800, 100)
    model.to("cpu")

    
    #summary(model, input_size=(1, 30, seq_len_list[-1]))
    
    print(torch.cuda.current_device())
    model_device = next(model.parameters()).device
    print("Model Device:", model_device)
    if True:
        train_x, valid_x = initData(seq_len=seq_len_list[-1], stride=100, batch_size=batch_size[1])
        
        train_time, train_total_loss , valid_total_loss    = train(model, train_x, valid_x, epochs[0], alphas[1])

        save_model(model, archID, seq_len_list[0], train_total_loss[-1], valid_total_loss[-1], train_time, interactive=False)
        #save_model(model, archID, seq_len_list[-1], 0, 0, train_time, interactive=False)
    if False:
        seq = seq_len_list[0]
        batch = batch_size[0]
        epoch = epochs[1]
        for alpha in alphas:
            train_x, valid_x = initData(seq_len=seq, stride=10, batch_size=batch)
            train_time, train_total_loss, valid_total_loss  = train(model, train_x, valid_x, epoch, alpha)
            save_model(model, archID, seq_len_list[0], train_total_loss[-1], valid_total_loss[-1], train_time, interactive=False)
