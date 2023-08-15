import torch 
import os
import pandas as pd

from datetime import datetime

import shutup 
shutup.please()

# Custom imports
from dataset import TimeseriesTensor 
from ae_model import Models

def main(model_id):
    # Hardcoded Data
    path_Cp_data = '../data/cp_data_true/AoA_0deg_Cp'
    path_results = "../training_results.csv"
    path_models = f'../trained_models/{model_id}.pt'

    path_pickeld =  '../data/pickeld/'
    folder = f"{model_id}-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path_original = path_pickeld + folder + "/original/"
    path_reconstructed = path_pickeld + folder + "/reconstructed/"
    os.mkdir(path_pickeld+folder)
    os.mkdir(path_original)
    os.mkdir(path_reconstructed)

    # the ususal train/validate/test split
    train_exp = [3,4,7,8,12,13,17,22,23,26,31,32,35,36,41,42,45,46,50,51,55,60,64,65,69,70,73,74,79,80,83,84,88, 89,93,98,99,102,107,108,111, 112]
    valid_exp = [16,27,54,61,92,103]
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]
 
    experiments = train_exp + valid_exp + test_exp

    # Load the Testdata
    df = pd.read_csv(open(path_results))
    seq_len = df[df["model_id"] == model_id]['window_size'].values[0]
    
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
        torch.save(test_x, path_original + filename)
        torch.save(test_x_hat, path_reconstructed + filename)

if __name__ == "__main__":
    main("CA5B:E21B:71ED:3A1C")
    print("Done\n")

