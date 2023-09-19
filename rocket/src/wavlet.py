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
from simple_term_menu import TerminalMenu
import sys
sys.path.append("../../aerolib")
import modelManagement as mm
import ploter as pl 
import dataset



# path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_Cp_data = '../../data/raw_data/aerosense/'
tensor = dataset.TensorLoaderCp(path_Cp_data, [5], skiprows=0)

fs = 100
print(f"XX: {tensor}")
sensor = 17
sig = torch.transpose(tensor[0], 0, 1)[sensor].numpy()
samples = len(sig)
t = [t1/(100*60) for t1 in range(samples)]
# and a range of scales to perform the transform



fig, ax = plt.subplots(1, 1, figsize=(10,2));  

ax.plot(t,sig)
ax.set
ax.set_title("Original Cp Signal")

# and the scaleogram
plt.tight_layout()
plt.show()

