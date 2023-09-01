import random
import os
from dataclasses import dataclass, fields, asdict
from simple_term_menu import TerminalMenu
# importing the module
import pandas as pd
	
@dataclass
class Model_entry:
    model_id: str
    arch_id: str = ""
    parameters: int = 0
    alpha: float = 0.0
    epochs: int = 0
    batch_size: int = 0
    activation: str = ""
    window_size: int = 0
    latent_channels: int = 0 
    latent_seq_len: int = 0
    train_loss: float = 0.0
    valid_loss: float = 0.0
    train_time: float = 0.0
    mse: float = 0.0


def _default_file() -> str:
    file = '../models.csv'
    if not os.path.isfile(file):
        _create_csv(file)
    
    return file 

def _generateModelID() -> str:
    hex_num  =  hex(random.randint(0, 16**16-1))[2:].upper().zfill(16)
    hex_num  =  hex_num[:4] + ":" + hex_num[4:8] + ":" + hex_num[8:12] + ":" + hex_num[12:]
    return hex_num


def loadDF() -> pd.DataFrame:
    file = _default_file()
    return pd.read_csv(file)

def _create_csv(file) -> None:
    fieldnames = [field.name for field in fields(Model_entry(''))]
    df = pd.DataFrame(columns=fieldnames)
    df.to_csv(file)


def saveModel(model: Model_entry) -> None:
    file = _default_file()
    df = pd.read_csv(file)
    df_new = pd.DataFrame([asdict(model)])
    df = pd.concat([df, df_new])
    df.to_csv(file, index=False)


def createNewModel():
    return Model_entry(_generateModelID())


def updateModel(model):
    file = _default_file()
    df = pd.read_csv(file)
    idx = df.index[df["model_id"]==model.model_id][0]
    df.iloc[idx] = asdict(model)
    df.to_csv(file, index=False)


def loadByID(idx: str) -> Model_entry:
    file = _default_file()
    df = pd.read_csv(file)
    entry = df.loc[df["model_id"]==idx].to_dict('index')
    return Model_entry(**entry.popitem()[1])


def modelChooser() -> Model_entry:
    file = _default_file()
    df = pd.read_csv(file)
    options = []
    model_ids = []
    for row in df.iterrows():
        x = row[1]
        model_ids.append(x['model_id'])
        options.append(f"{x['model_id']} - {x['window_size']} -> {x['latent_channels']} x {x['latent_seq_len']}")
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()

    model_id = model_ids[menu_entry_index]
    return loadByID(model_id)


