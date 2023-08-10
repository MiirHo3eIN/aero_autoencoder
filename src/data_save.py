import numpy as np 
import pandas as pd
# To save the trained model
import joblib
from csv import writer, DictWriter
import random
import os


MSE = lambda x, x_hat: np.mean(np.square(x - x_hat), axis = 1)


def generate_hexadecimal() -> str:
    hex_num  =  hex(random.randint(0, 16**16-1))[2:].upper().zfill(16)
    hex_num  =  hex_num[:4] + ":" + hex_num[4:8] + ":" + hex_num[8:12] + ":" + hex_num[12:]
    return hex_num

def is_file_exist(file_name: str) -> bool:
    return os.path.isfile(file_name)

def write_to_csv(data: dict) -> None: 
    write_dir = "../training_results.csv"
    if (is_file_exist(write_dir)): 
        append_to_csv(data)
    else: 
        create_csv(data)

def append_to_csv(data: dict) -> None:
    
    print("Appending to the training results csv file")
    print("++"*15)
    with open("../training_results.csv", "a") as FS:
        
        headers = list(data.keys())

        csv_dict_writer = DictWriter(FS, fieldnames = headers) 

        csv_dict_writer.writerow(data)

        FS.close()


def create_csv(data: dict) -> None:
    df = pd.DataFrame.from_dict(data, orient = "index").T.to_csv("../training_results.csv", header = True, index = False)
    print("Created the csv file is as follows:")
    print(df)
