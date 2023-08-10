from dataset import TimeSeriesDataset
from dataset import FeatureDataset
from torch.utils.data import DataLoader
from tsai.all import build_ts_model, create_rocket_features, ROCKET, MiniRocket
import tsai.all
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from tsai.models.MINIROCKET_Pytorch import *
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sktime.trasformation.panel.rocket import MiniRocketMultivariate
#######################################################################################

# User input

# path_Cp_data = 'C:/Users/Hiwi/Desktop/mvts_analysis/AoA_8deg_Cp'
# path_train_feat = 'C:/Users/Hiwi/Desktop/mvts_analysis/features_8deg/train_features_8deg'
# path_train_label = 'C:/Users/Hiwi/Desktop/mvts_analysis/features_8deg/train_labels_8deg'
# path_test_feat = 'C:/Users/Hiwi/Desktop/mvts_analysis/features_8deg/test_features_8deg'
# path_test_label = 'C:/Users/Hiwi/Desktop/mvts_analysis/features_8deg/test_labels_8deg'


# Amir's path
path_Cp_data = '/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_8deg_Cp'
path_train_feat = '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_8deg/train_features'
path_train_label = '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_8deg/train_labels'
path_test_feat = '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_8deg/test_features'
path_test_label = '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_8deg/test_labels'


# define training and test set based on design of experiment excel document

train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112]
test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

# for loading extracted features: set to False
create_features = False

# choose number of decision trees in random forest classifier

n_estimators = 800

#######################################################################################

numTrain = len(train_exp)
numTest = len(test_exp)

if create_features:

    # init the datasets
    train_set = TimeSeriesDataset(path_Cp_data,train_exp)
    test_set = TimeSeriesDataset(path_Cp_data,test_exp)
    # init the dataloaders
    train_loader = DataLoader(train_set, batch_size=numTrain*5, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=numTest*5, shuffle=False)

    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
    print(f'Using device: {device}')
    print('------------------------------------------')

    print("Start building the model...")
    # build the model from ts-ai
    model = build_ts_model(ROCKET, c_in=36, seq_len=2000, device=device)
    model.double()
    print("Model built!")
    print('------------------------------------------')
    print("Start generating the features...")
    # generate the rocket features
    mvts_train, label_train = create_rocket_features(train_loader, model)
    mvts_test, label_test = create_rocket_features(test_loader, model)
    print("Features generated!")
    print('------------------------------------------')
    #print(label_train[0].shape)
    #print(label_train[0])
    print('------------------------------------------')
    # save training features and labels
    for idx, tensor in enumerate(mvts_train):
        torch.save(tensor, f"{path_train_feat}/tensor_train{idx}_0_deg.pt")

    for idx, label in enumerate(label_train):
        torch.save(label, f"{path_train_label}/label_train{idx}_0_deg.pt")

    # save test features and labels
    for idx, tensor in enumerate(mvts_test):
        torch.save(tensor, f"{path_test_feat}/tensor_test{idx}_0_deg.pt")

    for idx, label in enumerate(label_test):
        torch.save(label, f"{path_test_label}/label_test{idx}_0_deg.pt")

else:

    # train loader
    train_features = FeatureDataset(path_train_feat)
    train_labels = FeatureDataset(path_train_label)

    print(train_features[0].shape)
    print(train_labels[0])    
    #exit()  
    test_features = FeatureDataset(path_test_feat)
    test_labels = FeatureDataset(path_test_label)

    train_load = DataLoader(train_features, batch_size=numTrain*5, shuffle=False)
    train_label_load = DataLoader(train_labels, batch_size=numTrain*5, shuffle=False)

    test_load = DataLoader(test_features, batch_size=numTest*5, shuffle=False)
    test_label_load = DataLoader(test_labels, batch_size=numTest*5, shuffle=False)

    for i in range(1):
        mvts_train=(next(iter(train_load)))
        label_train = (next(iter(train_label_load)))

        mvts_test=(next(iter(test_load)))
        label_test = (next(iter(test_label_load)))

    
label_train = (label_train.squeeze().detach().numpy())
mvts_train = np.nan_to_num(mvts_train.squeeze().detach().numpy(), nan=0.0)

label_test = (label_test.squeeze().detach().numpy())
mvts_test = np.nan_to_num(mvts_test.squeeze().detach().numpy(), nan=0.0)

# fit a Ridge regression classifier
ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17)) #, normalize=True)
ridge.fit(mvts_train, label_train.squeeze())

print('------------------------------------------')
print(f'alpha: {ridge.alpha_:.2E}  train: {ridge.score(mvts_train, label_train.squeeze()):.5f}  test: {ridge.score(mvts_test, label_test.squeeze()):.5f}')



# quantify classification accuracy
y_ridgeCV = ridge.predict(mvts_test)
cmatrix_ridge = confusion_matrix(label_test, y_ridgeCV)

plt.figure()
with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':14,
                        'ytick.labelsize':14}):
    ax = sns.heatmap(cmatrix_ridge/np.sum(cmatrix_ridge), annot=True, fmt='.1%',cmap='Blues', annot_kws={'size':14})
plt.savefig('Cmatrix_ridge_8_2.pdf')

print('------------------------------------------')
print('Classification details ridge classifier')
print('Confusion matrix\n\n\ ',cmatrix_ridge)
print(classification_report(label_test,y_ridgeCV))



print(label_test.shape)
# fit a random forest classifier 
rfc = RandomForestClassifier(n_estimators,max_features="sqrt")
rfc.fit(mvts_train, label_train)

#predict test results
y_rfc = rfc.predict(mvts_test)

# check model accuracy score
print('RFC model accuracy score with '+str(n_estimators)+' decision trees:{0:0.4f}'. format(accuracy_score(label_test, y_rfc)))

cmatrix_rfc = confusion_matrix(label_test, y_rfc)

plt.figure()
with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':14,
                        'ytick.labelsize':14}):
    ax = sns.heatmap(cmatrix_rfc/np.sum(cmatrix_rfc), annot=True, fmt='.1%',cmap='Blues', annot_kws={'size':14})
#            fmt='.2%', cmap='Blues')

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['healthy and 5 mm', '10mm and added mass', '15 and 20mm'])
#cm_display.plot()
#plt.show()


print('--------------------------------------------------------------')
print('Classification details random forest classifier with '+str(n_estimators)+' trees')
print('Confusion matrix\n\n\ ',cmatrix_rfc)
print(classification_report(label_test,y_rfc))
d=2