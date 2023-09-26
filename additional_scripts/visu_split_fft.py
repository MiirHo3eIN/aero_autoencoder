import torch 
import numpy as np 

import matplotlib.pyplot as plt
import shutup 
shutup.please()

import sys
sys.path.append("../aerolib")
import modelManagement as mm
import ploter as pl 
import experiments as exp 
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
experiments, label, desc,  wind, excitation = exp.chooseSetOfExperiments()
sensor = exp.chooseSensor()

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
N1, N2 = 1500, 8192
xs1 = [x*(100/N1) for x in range(int(N1/2)+1)]
xs2 = [x*(100/N2) for x in range(int(N2/2)+1)]

for e in experiments:

    data = dataset.TensorLoaderCp(path_Cp_data, [e], skiprows=0)[0]    
    x = data[sensor]
    x_hat_1 = np.abs(np.fft.rfft(x[:1500]))
    x_hat_2 = np.abs(np.fft.rfft(x[2500:(2500+8192)]))

    print(f"data: {data.shape}")
    print(f"X: {x.shape}")
    print(f"data: {data.shape}")
    desc = exp.Damage_Classes().ex2desc(e)
    ax[0].plot(xs1, x_hat_1, label=f"Exp: {e}")
    ax[1].plot(xs2, x_hat_2, label=f"Exp: {e} ")
    
pl.descAxis(ax[0], log=True)
pl.descAxis(ax[1], log=True)
ax[0].set_title(f"FFT of the first 15s")
ax[1].set_title(f"FFT of the rest")
    #fig.tight_layout()
fig.suptitle(f"Real FFT of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor: {sensor}")
plt.savefig(f"plot/split_fft_exp{experiments}_s{sensor}.png")
plt.show()
   
