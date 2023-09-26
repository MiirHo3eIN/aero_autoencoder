import torch 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shutup 
shutup.please()

import sys
sys.path.append("../aerolib")
import modelManagement as mm
import ploter as pl 
import experiments as exp 
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_raw_data = '../data/raw_data/aerosense/'
path = path_raw_data

experiments, label, desc,  wind, excitation = exp.chooseSetOfExperiments()
sensor = exp.chooseSensor()

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
for e in experiments:
    data = dataset.TensorLoaderRaw(path, [e])[0]    
    x = data[sensor]
    print(f"data: {data.shape}")
    print(f"X: {x.shape}")
    desc = exp.Damage_Classes().ex2desc(e)
    ax.plot(x, label=f"Exp: {e}")
    
pl.descAxis(ax)
fig.suptitle(f"Cp Timeseries of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
plt.savefig(f"plot/ts_exp{experiments}_s{sensor}.png")
plt.show()
