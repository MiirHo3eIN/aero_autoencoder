import torch 
import numpy as np
import pandas as pd

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
import experiments as damages

path_Cp_data = '../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_raw_data = '../data/raw_data/aerosense/'
path = path_raw_data
# path = path_Cp_data

class TensorLoaderCp():
    def __init__(self, path, experiments: list, skiprows=2500) -> None:
        self._path = path
        self._exp = experiments
        self._datasetlen = len(self._exp)
        self._skiprows = skiprows

    def __len__(self) -> int: 
        return self._datasetlen

    def __getitem__(self, idx: int) -> torch.Tensor:
        print("Tensor LoaderCp()")
        exp = self._exp[idx]
        print(f"Loading Experiment: {exp}")
        #del_cells = [0, 23]
        del_cells = []
        cols = np.arange(1, 39)
        print(f"Cols: {cols}")
        use_cols = np.delete(cols, del_cells)
        print(f"use cols: {use_cols}")
        filepath = self._path+f'/aoa_0deg_Exp_{exp:03}_aerosense.csv'
        df = pd.read_csv(open(filepath,'r'),
                         delimiter=',',
                         skiprows = self._skiprows,
                         # usecols = use_cols,
                         )   
        print(df.shape)

        print(df.head)
        print("\n\n")

        tensor = torch.tensor(df.values, dtype = torch.float32)
        return torch.transpose(tensor, 0, 1)

class TensorLoader():
    def __init__(self, path, experiments: list, cols: list, skiprows=2500) -> None:
        self._path = path
        self._exp = experiments
        self._datasetlen = len(self._exp)
        self._skiprows = skiprows
        self._cols = cols

    def __len__(self) -> int: 
        return self._datasetlen

    def __getitem__(self, idx: int) -> torch.Tensor:
        exp = self._exp[idx]
        print(f"Loading Experiment: {exp}")
        #del_cells = [0, 23]
        del_cells = []
        cols = np.arange(1, 39)
        print(f"Cols: {cols}")
        use_cols = np.delete(cols, del_cells)
        print(f"use cols: {use_cols}")
        filepath = self._path+f'/aoa_0deg_Exp_{exp:03}_aerosense.csv'
        df = pd.read_csv(open(filepath,'r'),
                         delimiter=',',
                         skiprows = self._skiprows,
                         # usecols = use_cols,
                         )   
        print(df.shape)

        print(df.head)
        print("\n\n")

        tensor = torch.tensor(df.values, dtype = torch.float32)
        return torch.transpose(tensor, 0, 1)
def plotDescAxis(ax, xLabel="Time [s]", yLabel="Cp"):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True)
    ax.legend()

if __name__ == "__main__":
    e = 3
    #sensor = chooseSensor()
    wind = damages.Damage_Classes().ex2wind(e)
    desc = damages.Damage_Classes().ex2desc(e)
    excitation = damages.Damage_Classes().ex2excit(e)
    data = TensorLoaderCp(path, [e], skiprows=3)
    data = data[0]
    print(data.shape)
    # for sensor in range(1, 39):
    for sensor in range(1, 2):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(data[sensor-1], label=f"Exp: {e}")
        plotDescAxis(ax)
        fig.suptitle(f"Cp Timeseries of the Experiments with {desc} \n  Windspeed: {wind} | Excitation: {excitation} | Sensor:{sensor}")
        plt.savefig(f"plot/dataloader_test_exp{e}_s{sensor}.png")
        plt.show()

