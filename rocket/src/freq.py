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


def plotDescAxis(ax, xLabel="Freq. [Hz]", yLabel="Cp"):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

# Plots only a single signal 
def plotCpFFT(exp, sensor, window):
    path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
    data = dataset.TimeseriesTensor(path_Cp_data, [exp], seq_len=2048, stride=2048)    
    x = data[window][sensor]
    x_hat = np.abs(np.fft.rfft(x))

    print(f"X_hat: {x_hat.shape}")
    print(f"X_hat: {type(x_hat[0])}")
    print(f"data: {data.shape}")
    desc = dataset.Damage_Classes.ex2desc(exp)
    x_scale = [x*(100/2048) for x in range(1024)]
    fig, ax = plt.subplots()
    ax.plot(x_scale, x_hat[1:], color = 'grey')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('CP')
    ax.grid(True)
    ax.set_title(f"Real FFT of sensor {sensor} without DC part (DC={x_hat[0]}) \n with {desc} | Exp {exp}")
    fig.tight_layout()
    plt.savefig(f"../plot/single_s{sensor}_exp{exp}_w{window}_seq2048.png")
    plt.show()

# plots multiple signals ontop of eachother
def plotFFT(exp, sensor, window):
    path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
    fig, ax = plt.subplots(1, 2)
    N = 4096
    x_scale = [x*(100/N) for x in range(int(N/2))]

    for e in exp:
        data = dataset.TimeseriesTensor(path_Cp_data, [e], seq_len=N, stride=N)    
        x = data[window][sensor]
        x_hat = np.abs(np.fft.rfft(x))

        print(f"X_hat: {x_hat.shape}")
        print(f"X_hat: {type(x_hat[0])}")
        print(f"data: {data.shape}")
        desc = dataset.Damage_Classes.ex2desc(e)
        ax[0].plot(x_scale, x_hat[1:], label=f"Exp: {e} | {desc} | Sensor: {sensor} ")
        ax[1].plot(x_scale[:512], x_hat[:512], label=f"Exp: {e} | {desc} ")
    
    plotDescAxis(ax[0])
    plotDescAxis(ax[1])
    ax[0].set_title(f"Real FFT of sensor {sensor} without DC part (DC={x_hat[0]})")
    ax[1].set_title(f"Zoomed in version of the left plot")
    #fig.tight_layout()
    plt.savefig(f"../plot/single_s{sensor}_exp{exp}_w{window}_seq2048.png")
    plt.show()



if __name__ == "__main__":
     
    # plotFFT(exp=[5, 43, 62, 81, 100], sensor=25, window=0) # wind: 10ms | exiting 1 Hz
    # plotFFT(exp=[12, 47, 66, 85, 104], sensor=17, window=0) # wind: 20m/s | exiting: 1 Hz
    plotFFT(exp=[3, 8, 13, 17, 24], sensor=17, window=0) # wind: 20m/s | exiting: 1 Hz
    
    # plotCpFFT(exp=5, sensor=17, window=0)
    # plotCpFFT(exp=43, sensor=17, window=0)
    # plotCpFFT(exp=62, sensor=17, window=0)
    # plotCpFFT(exp=81, sensor=17, window=0)
    # plotCpFFT(exp=100, sensor=17, window=0)
    
    # s=25
    # plotCpFFT(exp=5, sensor=s, window=0) # 0mm
    # plotCpFFT(exp=43, sensor=s, window=0) # 5mm
    # plotCpFFT(exp=62, sensor=s, window=0) # 10mm
    # plotCpFFT(exp=81, sensor=s, window=0) # 15mm
    # plotCpFFT(exp=100, sensor=s, window=0) # 20mm
