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

def computation(res: Results):
    def compute_psd(x, sample_rate=100):
        fft = np.fft.fft(x)
        power_spectrum = np.abs(fft) ** 2
        psd = power_spectrum / len(x)
        freqs = np.fft.fftfreq(len(x), 1 / sample_rate)
        return psd, freqs
     
    # load a data
    raw_data = dataset.TensorLoaderRaw(path_raw_data, [res.exp], skiprows=2500) # 2500 -> we want the staionary part of the data    

    # calculate fft
    N = res.fft_N
    fs = 100
    # xs = [x*(100/N) for x in range(int(N/2)+1)]
    ts = raw_data[0][res.sensor-1][:N]
    # print(ts.shape)
    psd, freq = compute_psd(ts)

    # http://blog.dddac.com/noise-floor-and-s-n-ratio-in-and-with-fft-graphs/
    noise1 = np.mean(psd[4000:5000])/(fs/N) 
    noise2 = np.mean(psd[4000:5000])/np.sqrt(fs/N) 
    plot_n1 = np.repeat(noise1, len(freq))
    plot_n2 = np.repeat(noise2, len(freq))


    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(freq, psd, label=f"Exp: {res.exp}")
    ax.plot(freq, plot_n1,  label=f"np.mean(psd[above 40hz])/(fs/N)")
    ax.plot(freq, plot_n2,  label=f"np.mean(psd[above 40hz])/np.sqrt(fs/N)")
    pl.descAxis(ax, xLabel="Frequency [Hz]", yLabel="Absolute Pressure", log=True )
    fig.suptitle(f"PSD of the staionary signal of Experiments with {dc.ex2desc(res.exp)} \n  Windspeed: {res.wind} | Excitation: {res.exci} | Sensor:{res.sensor}")
    # plt.savefig(f"plot/fft_exp{experiments}_s{sensor}.png")
    plt.show()


if __name__ == "__main__":
    res = Results()
    
    res.exp = 5
    res.wind = dc.ex2wind(res.exp)
    res.exci = dc.ex2excit(res.exp)

    res.sensor = 17

    computation(res)

    print(res)
