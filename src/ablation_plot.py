import torch 
import torch.nn as nn
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shutup 
shutup.please()

# Custom imports
from dataset import * 
from ablation_models import Model
from utils import updateMSE, modelChooser

# Hardcoded Data
path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_models = '../ablation_study/models/'

title_size = 20
label_size = 12

def loadData(md):
    # check if results exists
    file = f"../ablation_study/results/{md['model_id']}.pt"    
    if not os.path.isfile(file):
        print(f"Results not present for model {md['model_id']}")
    
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

    t = f'''Model ID: {md['model_id']} 
Arch ID: {md['arch_id']}
Parameters: {md['parameters']} 
Alpha: {md['alpha']}
Epochs: {md['epochs']} 
Batch Size: {md['batch_size']} 
Window Size: {md['window_size']}
Latent Channels: {md['latent_channels']}
Latent Sequence Size: {md['latent_seq_len']}

Classification:
Ridge accuracy: {data['ridge']['acc'][0]:.3}
Ridge precision: {data['ridge']['per'][0]:.3}

Random Forest accuracy: {data['rfc']['acc'][0]:.3}
Random Forest precision: {data['rfc']['per'][0]:.3}

'''
    ax.text(left, top, t,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

    ax.set_axis_off()


def plot(md, data):
    signals = data['signals']
    sensor = random.randint(1, 36)

    fig = plt.figure(figsize=(10, 20))

    gs = GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    timeseries(ax1, signals[:,:,300:500], sensor, title=f"Original and Reconstructed signals \n only Sensor {sensor} and  only a subsection is displayed [300:500]")

    ax2 = fig.add_subplot(gs[1, :-1])
    timeseries(ax2, signals[:, : , 400:450], sensor, title="")

    ax3 = fig.add_subplot(gs[1, 3])
    modelText(ax3, md, data)

    ax4 = fig.add_subplot(gs[-1, :2])
    confusionMatrix(ax4, data['ridge']['cm'][0], "Ridge")

    ax5 = fig.add_subplot(gs[-1, 2:])
    confusionMatrix(ax5, data['rfc']['cm'][0], "RFC")

    fig.set

    fig.suptitle("Model Evaluation", size=title_size*1.5)
    plt.show()


if __name__ == "__main__":
    md = modelChooser(file="../ablation_study/models.csv")
    data_dict = loadData(md)

    plot(md, data_dict)








