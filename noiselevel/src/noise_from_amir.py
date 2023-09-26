

from scipy import signal

import torch 
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shutup 
shutup.please()

import sys
sys.path.append("../../aerolib")
import modelManagement as mm
import ploter as pl 
import experiments as dmg 
dc = dmg.Damage_Classes()
import dataset

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_raw_data = '../../data/raw_data/aerosense/'

@dataclass
class Results:
    exp: int = 0 # Experiment Nr [1-114]
    wind: int = 0 # Windspeed either 0, 10 or 20
    exci: str = "1,00" # Excitation frequency at the tip, ["1,00" , "1,90"]
    sensor: int = 17
    datatype: str = "raw"  
    fft_N: int = 8192
    noiselevel: float = 0.0
    snr_mean: float = 0.0
    snr_1_9_Hz: float = 0.0

def psd_(time_ds, fs, log):
    
    f, Pxx_den = signal.welch(time_ds, fs = fs )
    if log == True:
        return {'freq': f, 'psd': np.log(Pxx_den)}
    return {'freq': f, 'psd': Pxx_den}


def noise_calculator(X, fs, full_detail = False): 

    psdxx = psd_(X, fs, log = False)
    freq = psdxx['freq'][:int(100/2)]
    psd_val = psdxx['psd'][:int(100/2)]
    
    noise = np.sqrt(np.mean(psd_val, axis = 0) )
    noise_average = np.mean(noise)*1e6
    if full_detail == True: 
            return { 'noise': noise, 'noise_average': noise_average  }
    return { 'noise': noise, 
              'noise_average': noise_average  }


def computation(res: Results):
    # load a data
    raw_data = dataset.TensorLoaderRaw(path_raw_data, [res.exp], skiprows=2500)[0] # 2500 -> we want the staionary part of the data    

    # calculate fft
    N = res.fft_N
    fs = 100
    # xs = [x*(100/N) for x in range(int(N/2)+1)]
    ts = raw_data[res.sensor-1]
    psdxx = psd_(ts, fs, False)
    freq = psdxx['freq']
    psd_val = psdxx['psd']
    noise = np.sqrt(np.mean(psd_val, axis = 0) )
    noise_average = np.mean(noise)*1e6
    noise_floor = np.repeat(noise, len(freq))
    print(noise)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(freq, psd_val, label=f"Exp: {res.exp}")
    ax.plot(freq, noise_floor,  label=f"noise")
    pl.descAxis(ax, xLabel="Frequency [Hz]", yLabel="Absolute Pressure", log=True )
    fig.suptitle(f"PSD of the staionary signal of Experiments with {dc.ex2desc(res.exp)} \n  Windspeed: {res.wind} | Excitation: {res.exci} | Sensor:{res.sensor}")
    # plt.savefig(f"plot/fft_exp{experiments}_s{sensor}.png")
    plt.show()

if __name__ == "__main__":
    res = Results()
    
    res.exp = 2
    res.wind = dc.ex2wind(res.exp)
    res.exci = dc.ex2excit(res.exp)

    res.sensor = 17

    computation(res)

    print(res)
