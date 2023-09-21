import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys 


from torch.utils.data import Dataset 

class RawDataset(Dataset): 

    def __init__(self, experiment) -> None:
        super().__init__()

        self.path = f"./raw_data/aoa_0deg_Exp_{experiment.zfill(3)}_aerosense.csv"