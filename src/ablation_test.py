import torch 
import torch.nn as nn
import pickle
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import shutup 
shutup.please()

# Custom imports
from dataset import * 
from ablation_models import Model
from utils import updateMSE, modelChooser

# Hardcoded Data
path_Cp_data = '../../data/cp_data/cp_data_true/AoA_0deg_Cp/'
path_models = '../ablation_study/models/'

# define training and test set based on design of experiment excel document
train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112]
test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]


def classifyRidge(train_x, train_labels, test_x, test_labels):

    print("\n\n"+"-"*50)
    print("[Ridge Classifier]")

    # fit a Ridge regression classifier
    ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17)) #, normalize=True)
    ridge.fit(train_x, train_labels)

    # quantify classification accuracy
    y_ridgeCV = ridge.predict(test_x)

    # asses performance
    cmatrix_ridge = confusion_matrix(test_labels, y_ridgeCV)
    acc = accuracy_score(test_labels,y_ridgeCV)
    pre = precision_score(test_labels, y_ridgeCV, average='macro')

    print(f'Classification accuracy: {acc}')
    print(f'Classification precision: {pre}')
    print('Confusion matrix:\n', cmatrix_ridge)

    return cmatrix_ridge, acc, pre




def classifyRFC(train_x, train_labels, test_x, test_labels, n_estimators=800):

    print("\n\n"+"-"*50)
    print("[RFC Classifier]")

    # fit a random forest classifier 
    rfc = RandomForestClassifier(n_estimators,max_features="sqrt")
    rfc.fit(train_x, train_labels)

    #predict test results
    y_rfc = rfc.predict(test_x)

    cmatrix_rfc = confusion_matrix(test_labels, y_rfc)
    acc = accuracy_score(test_labels,y_rfc)
    pre = precision_score(test_labels, y_rfc, average='macro')

    print('Random forest classifier used ' + str(n_estimators) + ' trees')
    print(f'Classification accuracy: {acc}')
    print(f'Classification precision: {pre}')
    print('Confusion matrix:\n', cmatrix_rfc)

    return cmatrix_rfc, acc, pre




def reconstruct(md, x): # md = dict containing all infos about a model
    
    # Load the AE 
    model = Model(md['arch_id'])
    model.load_state_dict(torch.load(path_models+f"{md['model_id']}.pt"))    

    # Run the Testdata through the Model
    with torch.no_grad():
        model.eval() 
        x_hat = model(x.float())
    
    return x_hat




def extractFeatures(md):

    print("\n\n"+"-"*50)
    print("[Extract Features]")
    print(f"Init Dataset ...")
    # init the datasets
    train_x, train_labels = TimeseriesSampledCpWithLabels(path_Cp_data, train_exp, 10, 800)
    test_x, test_labels = TimeseriesSampledCpWithLabels(path_Cp_data, test_exp, 10, 800)
    
    # print(train_x.shape)
    # print(train_labels.shape)
    print(f"Compress and Reconstruct ...")
    train_x = reconstruct(md, train_x)
    test_x = reconstruct(md, test_x)

    # print(train_x.shape)
    # print(train_labels.shape)
    print("Extract features with MiniRocket ...")
    # expects a tensor of shape [m, 36, n]
    # m = samples to transform
    # n = sample length (window length)

    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.fit(train_x.numpy())
    X_train_transform = minirocket_multi.transform(train_x.numpy())
    X_test_transform = minirocket_multi.transform(test_x.numpy())
    
    del train_x
    del test_x

    return X_train_transform, train_labels, X_test_transform, test_labels 




def calculateMSEandPlot(md):

    def criterion(x, x_hat):
        mse = nn.MSELoss()
        return mse(x.float(), x_hat.float())

    print("\n\n"+"-"*50)
    print("[Calculate MSE]")
    print(f"Compress and Reconstruct the testdata with model {md['model_id']}")

    # Load the Testdata
    x = TimeseriesTensor(path_Cp_data, test_exp, seq_len = md['window_size'], stride=20)    

    # Compress and reconstruct
    x_hat = reconstruct(md, x)

    # Calculate the metric
    output = criterion(x.float(), x_hat.float())
    print(f"Tested Model with MSE: {output:.5}")
    updateMSE(md['model_id'], output.item(), file="../ablation_study/models.csv")


    # return data of one Sequence for Plots
    # first entry is the original data second the reconstructed
    idx = random.randint(0, x.size(dim=0))
    return torch.cat((x[[idx]], x_hat[[idx]]))



if __name__ == "__main__":
    md = modelChooser(file="../ablation_study/models.csv")

    signals = calculateMSEandPlot(md)

    results = {'md': md,
               'signals': signals, 
            'ridge': {'cm': [],
                         'acc': [],
                         'per': [],
                         },
            'rfc': {'cm': [],
                       'acc': [],
                       'per': [],
                         },
               }

    for i in range(1):
        train_features, train_labels, test_features, test_labels = extractFeatures((md))
        
        cm, acc, per = classifyRidge(train_features, train_labels, test_features, test_labels) 
        results['ridge']['cm'].append(cm)
        results['ridge']['acc'].append(acc)
        results['ridge']['per'].append(per)

        cm, acc, per = classifyRFC(train_features, train_labels, test_features, test_labels) 
        results['rfc']['cm'].append(cm)
        results['rfc']['acc'].append(acc)
        results['rfc']['per'].append(per)
    
    # save dictionary to person_data.pkl file
    filename = f"{md['model_id']}.pt"    
    with open("../ablation_study/results/"+ filename, 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')

