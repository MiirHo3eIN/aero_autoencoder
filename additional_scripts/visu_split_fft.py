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

def plotDescAxis(ax, xLabel="Freq. [Hz]", yLabel="Cp"):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

def plotSingleFFT(experiments, label, desc,  wind, excitation, sensor):

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    N1, N2 = 1500, 8192
    xs1 = [x*(100/N1) for x in range(int(N1/2)+1)]
    xs2 = [x*(100/N2) for x in range(int(N2/2)+1)]

    for e in experiments:

        data = dataset.TensorLoaderCp(path_Cp_data, [e], skiprows=0)[0]    
        x = torch.transpose(data, 0, 1)[sensor]
        x_hat_1 = np.abs(np.fft.rfft(x[:1500]))
        x_hat_2 = np.abs(np.fft.rfft(x[2500:(2500+8192)]))

        print(f"data: {data.shape}")
        print(f"X: {x.shape}")
        print(f"data: {data.shape}")
        desc = exp.Damage_Classes().ex2desc(e)
        ax[0].plot(xs1, x_hat_1, label=f"Exp: {e}")
        ax[1].plot(xs2, x_hat_2, label=f"Exp: {e} ")
    
    plotDescAxis(ax[0])
    plotDescAxis(ax[1])
    ax[0].set_title(f"FFT of the first 15s")
    ax[1].set_title(f"FFT of the rest")
    #fig.tight_layout()
    fig.suptitle(f"Real FFT of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor: {sensor}")
    plt.savefig(f"plot/split_fft_exp{experiments}_s{sensor}.png")
    plt.show()

def plotAllFFT(experiments, label, desc,  wind, excitation):

    N1, N2 = 1500, 8192
    xs1 = [x*(100/N1) for x in range(int(N1/2)+1)]
    xs2 = [x*(100/N2) for x in range(int(N2/2)+1)]
    data = dataset.TensorLoaderCp(path_Cp_data, experiments, skiprows=0)    
    
    print(f"Exp: {experiments}")
    print(f"Data shape: {len(data)}")
    for s in range(0, 35):
        print(f"Sensor: {s}")
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        for i, e in enumerate(data):

            x = torch.transpose(e, 0, 1)
            print(f"X: {x.shape}")
            x = x[s]
            
            print(f"x[0]: {x[0]}")
            x_hat_1 = np.abs(np.fft.rfft(x[:1500]))
            x_hat_2 = np.abs(np.fft.rfft(x[2500:(2500+8192)]))

            desc = exp.Damage_Classes().ex2desc(experiments[i])
            ax[0].plot(xs1, x_hat_1, label=f"Exp: {experiments[i]}")
            ax[1].plot(xs2, x_hat_2, label=f"Exp: {experiments[i]} ")
    
        plotDescAxis(ax[0])
        plotDescAxis(ax[1])
        ax[0].set_title(f"FFT of the first 15s")
        ax[1].set_title(f"FFT of the rest")
        #fig.tight_layout()
        fig.suptitle(f"Real FFT of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{s+1}")
        plt.savefig(f"plot/split_fft_exp{experiments}_s{sensor}.png")
        plt.show()

def chooseSensor():
    opt = [x for x in range(1, 37)]
    opt.append("all")
    print(opt)
    print("\rChoose a Sensor:")
    optStr = map(str, opt)
    idx = TerminalMenu(optStr).show()
    return opt[idx]

def chooseSetOfExperiments():
    dc = exp.Damage_Classes()

    # choose label
    options = []
    for l, d in dc.labels:
        options.append(d)
    label = TerminalMenu(options).show()

    # choose wind
    opt = [None]
    for d in dc.windspeed:
        opt.append(d)
    optStr = map(str, opt)
    idx = TerminalMenu(optStr).show()
    windspeed = opt[idx]

    # choose excitation
    opt = [None]
    for d in dc.excitation:
        opt.append(d)
    optStr = map(str, opt)
    idx = TerminalMenu(optStr).show()
    excitation = opt[idx]

    desc = dc.labels[label][1]
    experiments = dc.filter(dc.labels[label][0], wind=windspeed, excitation=excitation)
    return experiments, label, desc, windspeed, excitation




if __name__ == "__main__":
    experiments, label, desc,  wind, excitation = chooseSetOfExperiments()
    window = 0
    sensor = chooseSensor()
    if sensor == "all":
        plotAllFFT(experiments, label, desc, wind, excitation)
    else:
        plotSingleFFT(experiments, label, desc,  wind, excitation, sensor)
