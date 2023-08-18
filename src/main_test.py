import torch 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shutup 
shutup.please()

# Custom imports
from dataset import * 
from ae_model import Models
from data_save import updateMSE


def criterion(x, x_hat):
    mse = nn.MSELoss()
    return mse(x.float(), x_hat.float())


def model_eval(model_id):
    
    # Hardcoded Data
    path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
    path_results = "../training_results.csv"
    path_models = f'../trained_models/{model_id}.pt'
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

    # Load the Testdata
    df = pd.read_csv(open(path_results))
    seq_len = df[df["model_id"] == model_id]['window_size'].values[0]
    latent = df[df["model_id"] == model_id]['latent_dim'].values[0]
    test_x = TimeseriesTensor(path_Cp_data, test_exp, seq_len = seq_len)    
    print(f"Test data loaded with shape: {test_x.shape}")

    # Load the Model
    model = Models.get(model_id)
    model.load_state_dict(torch.load(path_models))    

    # Run the Testdata through the Model
    with torch.no_grad():
        model.eval() 
        test_x_hat = model(test_x.float())
    
    # Calculate the metric
    output = criterion(test_x.float(), test_x_hat.float())
    print(f"Tested Model:{model_id} with MSE: {output:.5}")
    updateMSE(model_id, output.item())

    # Print the Results into a Plot
    sensor = 17
    title = f'Model: {model_id} | Sensor: {sensor} \n Sequence Length: {seq_len} | Latent Size: 18 x 200 \n MSE: {output}'
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
    # model_eval(tiny_cnn[3])
    # model_eval("8363:802A:3AC2:C596")
    model_eval("D9F2:0A27:942A:1DA0")
    # model_eval("4E09:4C41:54E0:9BF9")
    # model_eval("372D:4517:E7D3:34E9")
    # model_eval(cnn[4])
    # for model in cnn:
    #     model_eval(model)
 
    # for model in tiny_cnn:
        # model_eval(model)
