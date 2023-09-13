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

import scaleogram as scg 
import pywt

# choose default wavelet function for the entire notebook
scg.set_default_wavelet('cmor1-1.5')

# ugly hack because of a bug in the scg lib
COI_DEFAULTS = {'alpha':0.5,'hatch':'/',}


path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
tensor = dataset.TensorLoaderCp(path_Cp_data, [5], skiprows=0)

fs = 100
print(f"XX: {tensor}")
sensor = 17
sig = torch.transpose(tensor[0], 0, 1)[sensor].numpy()
samples = len(sig)
t = [t1/(100*60) for t1 in range(samples)]
# and a range of scales to perform the transform
scales = scg.periods2scales( np.arange(1, 100) )



fig, ax = plt.subplots(3, 1, figsize=(10,2));  

ax[0].plot(t,sig)
ax[0].set
ax[0].set_title("Original Cp Signal")

# and the scaleogram
scg.cws(sig, scales=scales, figsize=(6,2.3), coikw=COI_DEFAULTS)
plt.tight_layout()
plt.show()

