import torch 
from ae_model import CNN_AE

# read the model 
model_id = "7956:7BEB:31C5:19D5"

model = CNN_AE(c_in = 36)
models_path = f'../trained_models/{model_id}.pt'
model.load_state_dict(torch.load(models_path))    
model.eval()

dummy_input = torch.randn(10, 36, 200)

input_names = [ "input" ]
output_names = [ "output" ]



torch.onnx.export(model, dummy_input, f"../trained_models/{model_id}.onnx", verbose=False, input_names=input_names, output_names=output_names, export_params=True)

