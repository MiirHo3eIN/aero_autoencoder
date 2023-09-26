
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def seriesGrey(ax, signal, sensors, labels):
    for idx, sensor in enumerate(sensors):
        ax.plot(signal[sensor-1, :].detach().numpy(), color = 'grey', label = labels[idx])

def descAxis(ax, xLabel="Time [s]", yLabel="Cp", log=False):
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if log: ax.set_yscale('log')
    ax.grid(True)
    ax.legend()




