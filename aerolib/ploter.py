
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def seriesGrey(ax, signal, sensors, labels):
    for idx, sensor in enumerate(sensors):
        ax.plot(signal[sensor-1, :].detach().numpy(), color = 'grey', label = labels[idx])




