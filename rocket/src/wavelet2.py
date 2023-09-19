import torch 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import sys
sys.path.append("../../aerolib")
import dataset

t = np.linspace(-1, 1, 200, endpoint=False)

path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp'
tensor = dataset.TensorLoaderCp(path_Cp_data, [5], skiprows=0)

fs = 100
sensor = 17
sig = torch.transpose(tensor[0], 0, 1)[sensor][5000:5500].numpy()
t = [t1/fs for t1 in range(len(sig))]

widths = np.arange(1, 101)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
print(cwtmatr.shape)
cwtmatr_yflip = np.flipud(cwtmatr)
plt.imshow(cwtmatr_yflip, extent=[0, t[-1], 1, 101], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()
