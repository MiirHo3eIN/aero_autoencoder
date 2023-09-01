import torch
import torch.nn as nn

# These two should be merged into a Class 
def torch_eval(model, x_input: torch.Tensor) -> torch.Tensor:

    with torch.no_grad():
        model.eval() 
        x_hat = model(x_input.float())
    return x_hat

def data_compression(model_configs, train_x, test_x): 
    # Call the model to compress the data 

    model = ae_model.Model(model_configs.arch_id)
    
    model_id = model_configs.model_id
    model.load_state_dict(torch.load(f"{model_configs.path_models}{model_id}.pt"))  

    reconstructed_train = torch_eval(model, train_x)
    reconstructed_test = torch_eval(model, test_x)

    return reconstructed_train , reconstructed_test 

