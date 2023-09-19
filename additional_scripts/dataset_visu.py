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

def plotCpFFT():
    test_exp = [5]
    data = dataset.TimeseriesTensor(path_Cp_data, test_exp, seq_len=4096, stride=20)    
    x = data[0][1]
    print(f"X: {x.shape}")
    x_hat = np.abs(np.fft.rfft(x))
    print(f"X_hat: {x_hat.shape}")
    print(f"X_hat: {type(x_hat[0])}")

    fig, ax = plt.subplots()
    ax.plot(x_hat[1:], color = 'grey')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('CP')
    ax.grid(True)
    ax.set_title("Real FFT of one sensor without DC part")
    fig.tight_layout()
    plt.show()



def plotMeanCP():
    test_exp = [5]
    data = dataset.TimeseriesTensor(path_Cp_data, test_exp, seq_len=800, stride=800)    

    sensors = [x for x in range(36)]
    labels= [None for x in range(36)]
    x = torch.transpose(torch.mean(data, 2), 0, 1)
    print(x.shape)
    for window in data:
        for idx, s in enumerate(window):
            # print(type(s))
            # print(s.shape)
            m = np.mean(s.detach().numpy())
            print(f"Sensor {idx}: {m:.3}")
        break

    fig, ax = plt.subplots()
    pl.seriesGrey(ax, x, sensors, labels)
    ax.set_xlabel('Window')
    ax.set_ylabel('Mean(Cp)')
    ax.grid(True)
    ax.set_title("Mean of the Cp data over an experiment")
    fig.tight_layout()
    plt.show()
    
    z = x[:,0].tolist()
    print(type(z))
    fig, ax = plt.subplots()
    ax.bar(sensors, z)
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Mean(Cp)')
    ax.grid(True)
    ax.set_title("Mean of the Cp data over all sensors")
    fig.tight_layout()
    plt.show()



def plotCpTS():
    data = dataset.TimeseriesTensor(path_Cp_data, [5], seq_len=800, stride=20)    
    sensors = [x for x in range(36)]
    labels= [None for x in range(36)]

   fig, ax = plt.subplots()
    pl.seriesGrey(ax, data[0], sensors, labels)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Cp')
    ax.grid(True)
    ax.set_title("All channels of one window (Cp data)")
    fig.tight_layout()
    plt.show()
    
def plotDescAxis(ax, xLabel="Freq. [Hz]", yLabel="Cp"):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

# plots multiple signals ontop of eachother
def plotFFT():

    experiments, label, desc,  wind, excitation = chooseSetOfExperiments()
    window = 0
    sensor = chooseSensor()
    #sensor =17
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    N = 4096
    x_scale = [x*(100/N) for x in range(int(N/2))]

    for e in experiments:
        data = dataset.TimeseriesTensor(path_Cp_data, [e], seq_len=N, stride=N)    
        x = data[window][sensor]
        x_hat = np.abs(np.fft.rfft(x))

        print(f"data: {data.shape}")
        print(f"X: {x.shape}")
        print(f"X_hat: {x_hat.shape}")
        print(f"X_hat: {type(x_hat[0])}")
        print(f"data: {data.shape}")
        desc = exp.Damage_Classes().ex2desc(e)
        ax[0].plot(x_scale, x_hat[1:], label=f"Exp: {e}")
        ax[1].plot(x_scale[:256], x_hat[1:257], label=f"Exp: {e} ")
    
    plotDescAxis(ax[0])
    plotDescAxis(ax[1])
    ax[0].set_title(f"Full FFT (N={N}")
    ax[1].set_title(f"The First 6 Hz of the left Plot")
    #fig.tight_layout()
    fig.suptitle(f"Real FFT of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
    plt.savefig(f"plot/fft_exp{experiments}_s{sensor}_N{N}.png")
    plt.show()

def chooseSensor():
    opt = [x for x in range(1, 37)]
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
    plotFFT()
    exit()
    fnc = [plotCpTS, plotCpFFT, plotMeanCP, plotFFT]
    options = ["TS", "cpFFT", "meanCP", "FFT"]
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()
    
    fnc[menu_entry_index]()

