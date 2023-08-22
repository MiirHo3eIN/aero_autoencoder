from os.path import exists
import torch 
import os
import pandas as pd
from datetime import datetime

# Custom imports
from dataset import TimeseriesTensor 
from ae_model import Models

import shutup 
shutup.please()

###############################################################################
# User Input
# model_id = "CA5B:E21B:71ED:3A1C"
# model_id = "F06D:D524:BFD6:232E"
# model_id = "D86A:2185:C32B:7239"
# model_id = "A3B3:8C1F:43AC:7718"
# model_id = "B4AD:31CC:3620:B782"
# model_id = "7547:B8DA:C870:507A"
# model_id = "829C:AF16:5D58:E61C"
# model_id = "C019:A640:74EF:D675"	
model_id = "102E:5B5E:C956:FD77"

# Hardcoded Data
path_Cp_data = '../data/cp_data_true/AoA_0deg_Cp'
path_results = "../training_results.csv"
path_models = f'../trained_models/{model_id}.pt'

now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path_pickeld = f'../data/pickeld/{model_id}/{now}/'

path_o = path_pickeld + "/original/"
path_r = path_pickeld + "/reconstructed/"

os.makedirs(path_o, exist_ok=True)
os.makedirs(path_r, exist_ok=True)


############################################################################

def main():

    # the ususal train/validate/test split
    train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50,51,55,60,64,65,69,70,73,74,79,80,83,84,88, 89,93,98,99,102,107,108,111, 112]
    valid_exp = [16,27,54,61,92,103]
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]
 
    experiments = train_exp + valid_exp + test_exp

    # Load the Testdata
    df = pd.read_csv(open(path_results))
    seq_len = df[df["model_id"] == model_id ]['window_size'].values[0]
    
    # Load the Model
    model = Models.get(model_id)
    model.load_state_dict(torch.load(path_models))    

    for experiment in experiments:
        
        print(f"Experiment: {experiment}")
        # Load the data 
        test_x = TimeseriesTensor(path_Cp_data, [experiment], seq_len = seq_len)    

        print(f"Test X shape: {test_x.shape}")

        # Run the data through the model
        with torch.no_grad():
            model.eval() 
            test_x_hat = model(test_x.float())
    
        print(f"Test X Hat shape: {test_x_hat.shape}\n")

        # pickle reconstructed signal along with original signal
        filename = f"exp_{experiment:03}.pt"
        torch.save(test_x, path_o + filename)
        torch.save(test_x_hat, path_r + filename)

if __name__ == "__main__":
    main()
