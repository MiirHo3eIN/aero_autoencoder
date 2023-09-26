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
sys.path.append("../aerolib")
import modelManagement as mm
import ploter as pl 
import experiments as exp 
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'

experiments, label, desc,  wind, excitation = exp.chooseSetOfExperiments()
sensor = exp.chooseSensor(allSensors=False)

fig, ax = plt.subplots(2, 1, figsize=(20, 10))
N = 8192
xs = [x*(100/N) for x in range(int(N/2)+1)]

data = dataset.TensorLoaderCp(path_Cp_data, experiments, skiprows=0)    

    
        
for i, x in enumerate(data):
    ts = x[sensor-1][2500:(2500+8192)]
    x_hat = np.abs(np.fft.rfft(ts))

    ax[0].plot(ts, label=f"Exp: {experiments[i]}")
    ax[1].plot(xs, x_hat, label=f"Exp: {experiments[i]} ")
    
pl.descAxis(ax[0])
pl.descAxis(ax[1], xLabel="Frequency [Hz]", log=True )
ax[0].set_title(f"Timeseries")
ax[1].set_title(f"FFT with N=8192")
fig.suptitle(f"Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor: {sensor}")
plt.savefig(f"plot/ts_fft_exp{experiments}_s{sensor}.png")
plt.show()

