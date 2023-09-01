from dataclasses import dataclass

@dataclass
class Dataset_configs:
    path: str
    seq_len: int
    train_exp: list
    test_exp: list

@dataclass
class Model_configs:
    path_models: str
    model_id: str
    arch_id: str
    seq_len: int
