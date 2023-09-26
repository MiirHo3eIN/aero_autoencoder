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
import experiments as damages 
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_raw_data = '../data/raw_data/aerosense/'

exp, label, desc,  wind, excitation = damages.chooseSetOfExperiments()
sensor = damages.chooseSensor(allSensors=False)

plt.figure().suptitle(f"Compare TS of Raw vs. CP with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")

ax1 = plt.subplot(2, 1, 1)
raw = dataset.TensorLoaderRaw(path_raw_data, exp, skiprows=203)    
for i, e in enumerate(raw):
    ax1.plot(e[sensor], label=f"Exp: {exp[i]}")
pl.descAxis(ax1, yLabel="Absolute Pressure")

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
cp = dataset.TensorLoaderCp38(path_Cp_data, exp, skiprows=3)    
for i, e in enumerate(cp):
    ax2.plot(e[sensor], label=f"Exp: {exp[i]}")
pl.descAxis(ax2)


plt.savefig(f"plot/cmp_ts_exp{exp}_s{sensor}.png")
plt.show()
sys.path.append("../aerolib")
