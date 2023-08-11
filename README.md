# Data Compression Techniques for Surface Pressure of Wind Turbines

This project is intended to compress the pressure sensors data of a wind Turbine at the edge near the sensors. It is a Multivariate Time Series data with 36 channels input. Deploying CNNs with residual blocks are used in this part of the project to obtain a reconstructed signal which can be used to detect anomalies over the blade.  

![Logo](plots/encoder_decoder.png)


## Requirements

The python environment that we used to developed all the codes and test the model is listed in the following file. It is highly recommended to install all the requirements before starting the project. 
You can install all the requirements by running the following command in terminal. 
```bash
pip install -r requirements.txt
```

## Deployment

The models are described in the ```ae_models.py``` scripts.

To train the models run the following in the ``` src ``` folder

```bash
  python main_train.py 
```
The trained models are stored in the `trained_models` folder. A 16 digit random number is assigned to each model where the hyper-parameters of the model are stored in the ```training_results.csv``` file. 

To test the models run the following in ```src``` folder

```bash 
    python main_test.py
```
Notice to adjust hyper-paramters of the model before running the script to be coherent with the trained model that you would like to choose.  

## Contact 

To access data, please contact me via [Mail](amirhossein.moallem2@unibo.it)


## Project
```bash
├── LICENSE
├── plots
├── README.md
├── src
│   ├── ae_model.py
│   ├── ae_model_test.py
│   ├── data_save.py
│   ├── dataset_ae.py
│   ├── input_shape_trial.py
│   ├── main_test.py
│   ├── main_train.py
│   └── pytorch_to_onnx.py
└── trained_models
```
