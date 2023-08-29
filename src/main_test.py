import torch 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shutup 
shutup.please()

# Custom imports
from dataset import * 
from ae_model import Model
from utils import updateMSE, modelChooser


def criterion(x, x_hat):
    mse = nn.MSELoss()
    return mse(x.float(), x_hat.float())


def model_eval(md): # md = dict containing all infos about a model
    
    # Hardcoded Data
    path_Cp_data = '../data/cp_data/AoA_0deg_Cp'
    path_models = f"../trained_models/{md['model_id']}.pt"
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

    # Load the Testdata
    # seq_len = md['window_size']
    seq_len = 800
    test_x = TimeseriesTensor(path_Cp_data, test_exp, seq_len=seq_len, stride=20)    
    print(f"Test data loaded with shape: {test_x.shape}")

    # Load the Model
    model = Model(md['arch_id'])
    model.load_state_dict(torch.load(path_models))    

    # Run the Testdata through the Model
    with torch.no_grad():
        model.eval() 
        test_x_hat = model(test_x.float())
    
    # Calculate the metric
    output = criterion(test_x.float(), test_x_hat.float())
    print(f"Tested Model: {md['model_id']} with MSE: {output:.5}")
    updateMSE(md['model_id'], output.item())
    # Print the Results into a Plot
    
    cf = (36*md['window_size'])/(md['latent_channels'] * md['latent_seq_len'])
    # cf =8 
    sensor = 2
    sup_title = f"Model: {md['model_id']}"
    infos = f"Architecture: {md['arch_id']} | Sensor: {sensor} \n Seq. Length: {seq_len} | Latent Size: {md['latent_channels']} x {seq_len} \n MSE: {output:.03} | Compression F.: {cf}"
    with sns.plotting_context("poster"):
        sns.set(rc={'figure.figsize':(30,8.27)})
        plt.figure()
        plt.plot(test_x[0, sensor-1, :].detach().numpy(), color = 'green', label = 'original')
        plt.plot(test_x_hat[0, sensor-1, :], color = 'red', label = 'reconstructed signal')
        plt.legend()
        plt.suptitle(sup_title)
        plt.title(infos)
        plt.xlabel("Samples")
        plt.ylabel("Cp")
        plt.savefig(f"../plots/tests/test_{md['model_id']}.png")
        plt.show()

     




if __name__ == "__main__":
    model_dict = modelChooser()
    model_eval(model_dict)
