import torch 
import numpy as np

import matplotlib.pyplot as plt
import shutup 
shutup.please()

# Custom imports
import os
import sys
sys.path.append("../aerolib")
import modelManagement as mm
import ploter as pl 
import experiments as exp 
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
experiments, label, desc,  wind, excitation = exp.chooseSetOfExperiments()
sensor = exp.chooseSensor()

data = dataset.TensorLoaderCp(path_Cp_data, experiments, skiprows=0)    

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
N = 4096
xs = [x*(100/N) for x in range(int(N/2)+1)]

for i, x in enumerate(data):
    ts = x[sensor-1][2500:(2500+N)]
    x_hat = np.abs(np.fft.rfft(ts))
    ax.plot(xs, x_hat, label=f"Exp: {experiments[i]}")
    
pl.descAxis(ax, xLabel="Frequency [Hz]", log=True )
fig.suptitle(f"Cp (staionary) FFT of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
plt.savefig(f"plot/fft_exp{experiments}_s{sensor}.png")
plt.show()
