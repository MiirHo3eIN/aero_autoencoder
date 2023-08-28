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
from utils import updateMSE, modelChooser, selectModels, loadFileToDf

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


def plotSingleModel(md, data):
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

    fig.suptitle("Model Evaluation", size=title_size*1.5)
    plt.show()

def searcModel(df, a, c, s):
    return df.loc[(df['alpha'] == a) \
            & (df['latent_channels'] == c) \
            & (df['latent_seq_len'] == s)]\
            ['model_id'].values[0]

def getModelDict(df, model_id):
    idx = df.index[df['model_id'] == model_id].to_list()[0]
    return df.iloc[idx].to_dict()

def plotXY(ax, x, y, alphas):
    
    for idx, y_ in enumerate(y):
        ax.plot(x, y_, label=f"alpha = {alphas[idx]}")
    ax.legend()
    ax.xaxis.set_ticks(x)

def plotAblationSeq():


    file="../ablation_study/models.csv"
    df = loadFileToDf(file)
    
    # the used sequence lengths
    s = [25, 50, 100, 200, 400]
    alphas = [0.3, 0.5, 0.9]
    # group by alpha
    acc = [[], [], []]
    pre = [[], [], []]
    for idx, a in enumerate(alphas):
        for seq in s:
            m_id = searcModel(df, a=a, s=seq, c=36)
            md = getModelDict(df, m_id) 
            data = loadData(md)
            acc[idx].append(data['ridge']['acc'][0])
            pre[idx].append(data['ridge']['per'][0])
    # now we have a sorted list with model dicts

    fig = plt.figure(figsize=(10, 20))

    gs = GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    plotXY(ax1, s, acc, alphas)
    ax1.set_title("Accuracy (Rocket+Ridge Classifier")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Latent Space Sequence Length")

    ax2 = fig.add_subplot(gs[1,0])
    plotXY(ax2, s, pre, alphas)
    ax2.set_title("Precision Rocket+Rige Classifier")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("Latent Space Sequence Length")
    
    title = ''' Ablation Study:
Coparison of the latentspace sequence length and Classification
All models have 36 channels in the latent Space
Classier is trained on the reconstructed data
    '''

    fig.suptitle(title)
    plt.show()
    

def plotAblationCh():

    file="../ablation_study/models.csv"
    df = loadFileToDf(file)
    
    # the used sequence lengths
    ch = [1, 2, 5, 9, 18]
    alphas = [0.3, 0.5, 0.9]
    # group by alpha
    acc = [[], [], []]
    pre = [[], [], []]
    for idx, a in enumerate(alphas):
        for c in ch:
            m_id = searcModel(df, a=a, s=800, c=c)
            md = getModelDict(df, m_id) 
            data = loadData(md)
            acc[idx].append(data['ridge']['acc'][0])
            pre[idx].append(data['ridge']['per'][0])
    # now we have a sorted list with model dicts

    fig = plt.figure(figsize=(10, 15), layout='tight')

    gs = GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    plotXY(ax1, ch, acc, alphas)
    ax1.set_title("Accuracy")
    ax1.set_ylabel("Accuracy of the Ridge Classifier")
    ax1.set_xlabel("# of Latent Sapce Channels (Sequence Length = 800)")

    ax2 = fig.add_subplot(gs[1,0])
    plotXY(ax2, ch, pre, alphas)
    ax2.set_title("Precision")
    ax2.set_ylabel("Precision of the Ridge Classifier")
    ax2.set_xlabel("# of Latent Sapce Channels (Sequence Length = 800)")
    
    title = ''' Ablation Study:
Influence of the Latentspace Size of the Autoencoder on the Classifier
For every tick on the x axies 3 models were trianed with different alphas.
Alpha represents the ratio between L1 and MSE in the loss function. 
The data is compressed and subsequently reconstructed. Pipeline:
Original -> Encode -> Latent -> Decode -> Reconstructed -> Rocket -> Feature Map -> Rige Classifier '''

    fig.suptitle(title)
    plt.show()




if __name__ == "__main__":
    md = modelChooser(file="../ablation_study/models.csv")
    data_dict = loadData(md)

    plotSingleModel(md, data_dict)

    # plotAblationSeq()
    # plotAblationCh()









