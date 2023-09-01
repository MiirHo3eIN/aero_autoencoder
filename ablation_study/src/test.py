import torch 
import torch.nn as nn
import pickle

import random
import sys
sys.path.append("../../aerolib")
import modelManagement as mm
import dataset
import classifiers 

import shutup 
shutup.please()

# Custom imports
from models import Model

# Hardcoded Data
path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
path_models = '../models/'

# define training and test set based on design of experiment excel document
train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112]
test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]


def reconstruct(md, x): # md = dict containing all infos about a model
    
    # Load the AE 
    model = Model(md.arch_id)
    model.load_state_dict(torch.load(path_models+f"{md.model_id}.pt"))    

    # Run the Testdata through the Model
    with torch.no_grad():
        model.eval() 
        x_hat = model(x.float())
    
    return x_hat


def calculateMSEandPlot(md):

    def criterion(x, x_hat):
        mse = nn.MSELoss()
        return mse(x.float(), x_hat.float())

    print("\n\n"+"-"*50)
    print("[Calculate MSE]")
    print(f"Compress and Reconstruct the testdata with model {md.model_id}")

    # Load the Testdata
    x = dataset.TimeseriesTensor(path_Cp_data, test_exp, seq_len=md.window_size, stride=20)    

    # Compress and reconstruct
    x_hat = reconstruct(md, x)

    # Calculate the metric
    output = criterion(x.float(), x_hat.float()).item()
    print(f"Tested Model with MSE: {output:.5}")
    md.mse = output
    print(md)
    mm.updateModel(md)


    # return data of one Sequence for Plots
    # first entry is the original data second the reconstructed
    idx = random.randint(0, x.size(dim=0))
    return torch.cat((x[[idx]], x_hat[[idx]]))



if __name__ == "__main__":
    md = mm.modelChooser()

    signals = calculateMSEandPlot(md)

    results = {'md': md,
               'signals': signals, 
            'ridge': {'cm': [],
                         'acc': [],
                         'per': [],
                         },
            'rfc': {'cm': [],
                       'acc': [],
                       'per': [],
                         },
               }

    for i in range(1):
        
        print("-"*50)
        print("[Extract Features]")
        print(f"Init Dataset ...")
        train_x, train_labels = dataset.TimeseriesSampledCpWithLabels(path_Cp_data, train_exp, 10, 800)
        test_x, test_labels = dataset.TimeseriesSampledCpWithLabels(path_Cp_data, test_exp, 10, 800)
    
        print(f"Compress and Reconstruct ...")
        train_x = reconstruct(md, train_x)
        test_x = reconstruct(md, test_x)

        print("Extract features with MiniRocket ...")
        train_features, test_features = classifiers.rocket_feature_extraction(train_x, test_x)
    
        del train_x
        del test_x
        print("\n\n")


        print("-"*50)
        print("[Ridge Classifier]")
    
        cm_ridge, acc_ridge, pre_ridge = classifiers.ridge_classifier(train_features, train_labels, test_features, test_labels)
        results['ridge']['cm'].append(cm_ridge)
        results['ridge']['acc'].append(acc_ridge)
        results['ridge']['per'].append(pre_ridge)

        print(f'Classification accuracy: {acc_ridge}')
        print(f'Classification precision: {pre_ridge}')
        print('Confusion matrix:\n', cm_ridge)
        print('\n\n')

        print("-"*50)
        print("[RFC Classifier]")

        cm_rfc, acc_rfc, pre_rfc = classifiers.random_forest_classifier(train_features, train_labels, test_features, test_labels)
        results['rfc']['cm'].append(cm_rfc)
        results['rfc']['acc'].append(acc_rfc)
        results['rfc']['per'].append(pre_rfc)

        print('Random forest classifier used 800 trees')
        print(f'Classification accuracy: {acc_rfc}')
        print(f'Classification precision: {pre_rfc}')
        print('Confusion matrix:\n', cm_rfc)
        print('\n\n')
    
    # save dictionary to person_data.pkl file
    filename = f"{md.model_id}.pt"    
    with open("../results/"+ filename, 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')

