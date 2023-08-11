# Data Compression Techniques for Surface Pressure of Wind Turbines

This project is intended to compress the pressure sensors data of a wind Turbine at the edge near the sensors. It is a Multivariate Time Series data with 36 channels input. Deploying CNNs with residual blocks are used in this part of the project to obtain a reconstructed signal which can be used to detect anomalies over the blade.  

![Logo](plots/encoder_decoder.png)


## Documentation

TO DO 
[Documentation](https://linktodocumentation)


## Deployment

To deploy this project run the following in the ``` src ``` folder

```bash
  python main.py 
```

## Project 

```bash
├── data
├── LICENSE
├── README.md
└── src
    ├── dataset.py
    ├── feature_classnew.py
    └── main.py


```

## Contact 

for data access contact me via [Mail](amirhossein.moallem2@unibo.it)
## Directory Tree
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
│   ├── pytorch_to_onnx.py
│   ├── test_reconstructed.py
│   └── train_reconstructed.py
└── trained_models
```
