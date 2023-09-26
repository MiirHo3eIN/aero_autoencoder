import torch 

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

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
for e in experiments:
    data = dataset.TensorLoaderCp(path_Cp_data, [e], skiprows=0)[0]    
    ax.plot(data[sensor-1], label=f"Exp: {e}")
    
pl.descAxis(ax)
fig.suptitle(f"Cp Timeseries of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
plt.savefig(f"plot/ts_exp{experiments}_s{sensor}.png")
plt.show()

