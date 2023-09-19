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

def plotDescAxis(ax, xLabel="Time [s]", yLabel="Cp"):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True)
    ax.legend()

def plotTS():

    experiments, label, desc,  wind, excitation = chooseSetOfExperiments()
    window = 0
    sensor = chooseSensor()
    #sensor =17
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for e in experiments:
        data = dataset.TensorLoaderCp(path_Cp_data, [e], skiprows=0)[0]    
        x = torch.transpose(data, 0, 1)[sensor]
        print(f"data: {data.shape}")
        print(f"X: {x.shape}")
        desc = exp.Damage_Classes().ex2desc(e)
        ax.plot(x, label=f"Exp: {e}")
    
    plotDescAxis(ax)
    #fig.tight_layout()
    fig.suptitle(f"Cp Timeseries of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
    plt.savefig(f"plot/ts_exp{experiments}_s{sensor}.png")
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
    plotTS()
