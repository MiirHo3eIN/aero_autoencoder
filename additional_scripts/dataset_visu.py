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
from models import Model


def plotCpFFT():
    path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
    test_exp = [5]
    data = dataset.TimeseriesTensor(path_Cp_data, test_exp, seq_len=800, stride=20)    
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
    path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
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

    path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
    test_exp = [5]
    data = dataset.TimeseriesTensor(path_Cp_data, test_exp, seq_len=800, stride=20)    
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
    







if __name__ == "__main__":
    fnc = [plotCpTS, plotCpFFT, plotMeanCP]
    options = ["TS", "FFT", "meanCP"]
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()
    
    fnc[menu_entry_index]()

