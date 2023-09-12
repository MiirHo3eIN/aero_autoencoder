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
import sys
sys.path.append("../../aerolib")
import modelManagement as mm
import dataset
import classifiers 
from models import Model

# Hardcoded Data
path_Cp_data = '../../data/cp_data/AoA_0deg_Cp/'
path_models = '../models/'

title_size = 20
label_size = 12

def loadData(md):
    # check if results exists
    file = f"../results/{md.model_id}.pt"    
    if not os.path.isfile(file):
        print(f"Results not present for model {md.model_id}")
    
    # Read dictionary pkl file
    with open(file, 'rb') as fp:
        data = pickle.load(fp)

    return data

def timeseries(ax, signals, sensor, title):
    ax.plot(signals[0, sensor-1, :].detach().numpy(), color = 'green', label = 'original')
    ax.plot(signals[1, sensor-1, :], color = 'red', label = 'reconstructed signal')
    ax.legend()
    ax.set_title(title, size=title_size)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Cp")

def confusionMatrix(ax, cm, title):
    with plt.style.context({'axes.labelsize':title_size,
        'xtick.labelsize':label_size,
        'ytick.labelsize':label_size}):

        ax1 = sns.heatmap(cm/np.sum(cm), ax=ax, annot=True, fmt='.0%',cmap='Blues', annot_kws={'size':label_size})
        ax1.set_title(title)

def modelText(ax, md, data):
    # Build a rectangle in axes coords
    left, width = 0.0, 1.0
    bottom, height = 0.0, 1.0
    right = left + width
    top = bottom + height

    t = f'''Model ID: {md.model_id} 
Arch ID: {md.arch_id}
Parameters: {md.parameters} 
Alpha: {md.alpha}
Epochs: {md.epochs} 
Batch Size: {md.batch_size} 
Window Size: {md.window_size}
Latent Channels: {md.latent_channels}
Latent Sequence Size: {md.latent_seq_len}

Classification:
Ridge accuracy: {data['ridge']['acc']:.3}
Ridge precision: {data['ridge']['per']:.3}

Random Forest accuracy: {data['rfc']['acc']:.3}
Random Forest precision: {data['rfc']['per']:.3}

'''
    ax.text(left, top, t,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

    ax.set_axis_off()


def plot(md):
    data = loadData(md)
    signals = data['random_signal']
    sensor = random.randint(1, 36)

    fig = plt.figure(figsize=(22, 22))

    gs = GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    timeseries(ax1, signals[:,:,300:500], sensor, title=f"Original and Reconstructed signals \n only Sensor {sensor} and  only a subsection is displayed [300:500]")

    ax2 = fig.add_subplot(gs[1, :-1])
    timeseries(ax2, signals[:, : , 400:450], sensor, title="")

    ax3 = fig.add_subplot(gs[1, 2])
    plotMSE(ax3, data)

    ax4 = fig.add_subplot(gs[-1, 0])
    confusionMatrix(ax4, data['ridge']['cm'], "Ridge")

    ax5 = fig.add_subplot(gs[-1, 1])
    confusionMatrix(ax5, data['rfc']['cm'], "RFC")

    ax6 = fig.add_subplot(gs[-1, 2])
    modelText(ax6, md, data)
    
    fig.suptitle("Model Evaluation", size=title_size*1.5)

    # plt.tight_layout()
    plt.savefig(f"../plots/test_{md.model_id}.png")
    plt.show()

def plotMSE(ax, data):

    y = data["msePerSensor"]
    x = [i for i in range(1,37)]

    ax.bar(x, y)  
    current_values = ax.get_yticks()
    print(current_values)
    # using format string '{:.0f}' here but you can choose others
    ax.set_yticklabels(['{:,.02}'.format(x*1000) for x in current_values])
    ax.set_xlabel('Sensor')  
    ax.set_ylabel("MSE in $^o/_{oo}$")
    ax.set_title(f"MSE per sensor")



if __name__ == "__main__":
    md = mm.modelChooser()
    plot(md)











