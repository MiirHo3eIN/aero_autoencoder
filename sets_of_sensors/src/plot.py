import torch 
import numpy as np
import torch.nn as nn
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shutup 
shutup.please()

# Custom imports
import os
import sys
sys.path.append("../../aerolib")
import modelManagement as mm
import dataset
import classifiers 
from models import Model

# Hardcoded Data
path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
path_models = '../models/'

title_size = 20
label_size = 12

def loadData(md):
    # check if results exists
    file = f"../results/{md.model_id}.pt"    
    if not os.path.isfile(file):
        print(f"Results not present for model {md.model_id}")
    
    # Read dictionary pkl file
    with open(file, 'rb') as fp:
        data = pickle.load(fp)

    return data

def plotMSE(md, data):

    y = data["msePerSensor"]
    x = [i for i in range(1,37)]

    fig, ax = plt.subplots()
    ax.bar(x, y)  
    ax.set_ylabel('MSE')
    ax.set_title(f"MSE per sensor \n MSE Overall = {md.mse}")
    plt.show()



if __name__ == "__main__":
    md = mm.modelChooser()
    data_dict = loadData(md)
    plotMSE(md, data_dict)











