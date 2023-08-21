import numpy as np 
import pandas as pd
# To save the trained model
from csv import DictWriter
import random
import os
from simple_term_menu import TerminalMenu


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


def updateMSE(model_id, mse):
    df = pd.read_csv("../training_results.csv")
    # check if the mse columns already exists
    if not 'mse' in df.columns:
        df['mse'] = pd.Series()
    
    df.loc[df["model_id"]==model_id, "mse"] = mse
    df.to_csv("../training_results.csv", index=False)

def modelChooser():
    df = pd.read_csv("../training_results.csv")
    options = []
    model_ids = []
    for row in df.iterrows():
        x = row[1]
        model_ids.append(x['model_id'])
        options.append(f"{x['model_id']} - {x['window_size']} -> {x['latent_channels']} x {x['latent_seq_len']}")
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()

    model_id = model_ids[menu_entry_index]
    return df.loc[df['model_id'] == model_id].to_dict(orient='index')[menu_entry_index]

def checkNumber(string: str) -> int:
    integer = int(string)
    if integer in range(0, 10000):
        return integer
    raise Exception()
if __name__ == "__main__":
    
    # updateMSE("CA5B:E21B:71ED:3A1C", 0.00003)
    # print(modelChooser())
    
    answer = input("Do you want to save the Model? (y/n): ")
    if answer == "n": exit()

    answer = input("Please enter the latent channels? [0-100]: ")
    l_channels = checkNumber(answer)
    
    answer = input("Please enter the latent sequence length? [0-10000]: ")
    l_seq_len = checkNumber(answer)

    model_number = generate_hexadecimal()

    print("-"*50)
    print(f"Saving the model: {model_number}")

    
