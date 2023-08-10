from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sktime.transformations.panel.rocket import MiniRocketMultivariate

#######################################################################################

# User input

path_Cp_data = '/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_0deg_Cp'
path_saved_features =  '/home/miir_ho3ein/project/aerosense_CAD/rocket_features/features_0deg/'

# define training and test set based on design of experiment excel document

train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112]
test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

# for loading extracted features: set to False
create_features = True

# choose number of decision trees in random forest classifier

n_estimators = 800

#######################################################################################

numTrain = len(train_exp)
numTest = len(test_exp)

if create_features:

    print('------------------------------------------')
    print("Init Dataset and Dataloader")
    # init the datasets
    train_set = TimeSeriesDataset(path_Cp_data,train_exp)
    test_set = TimeSeriesDataset(path_Cp_data,test_exp)

    # init the dataloaders
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    print('------------------------------------------')
    print("Extract features with MiniRocket")
    X_train, labels_train = next(iter(train_loader))
    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.fit(X_train.numpy())
    X_train_transform = minirocket_multi.transform(X_train.numpy())

    X_test, labels_test = next(iter(test_loader))
    X_test_transform = minirocket_multi.transform(X_test.numpy())
    
    print('------------------------------------------')
    print("Save features")
    # save training features and labels
    np.save(path_saved_features+'train_features.npy', X_train_transform)
    np.save(path_saved_features+'train_labels.npy', labels_train.numpy())
    np.save(path_saved_features+'test_features.npy', X_test_transform)
    np.save(path_saved_features+'test_labels.npy', labels_test.numpy())
else:
    # train loader
    X_train_transform = np.load(path_saved_features+'train_features.npy')
    labels_train = np.load(path_saved_features+'train_labels.npy')
    X_test_transform = np.load(path_saved_features+'test_features.npy')
    labels_test = np.load(path_saved_features+'test_labels.npy')

# fit a Ridge regression classifier
ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17)) #, normalize=True)
ridge.fit(X_train_transform, labels_train)

# quantify classification accuracy
y_ridgeCV = ridge.predict(X_test_transform)
print('RFC model accuracy score with '+str(n_estimators)+' decision trees:{0:0.4f}'. format(accuracy_score(labels_test, y_ridgeCV)))
cmatrix_ridge = confusion_matrix(labels_test, y_ridgeCV)

plt.figure()
with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':14,
                        'ytick.labelsize':14}):
    ax = sns.heatmap(cmatrix_ridge/np.sum(cmatrix_ridge), annot=True, fmt='.1%',cmap='Blues', annot_kws={'size':14})
plt.show()

print('------------------------------------------')
print('Classification details ridge classifier')
print('Confusion matrix\n\n\ ',cmatrix_ridge)
print(classification_report(labels_test,y_ridgeCV))

# fit a random forest classifier 
rfc = RandomForestClassifier(n_estimators,max_features="sqrt")
rfc.fit(X_train_transform, labels_train)

#predict test results
y_rfc = rfc.predict(X_test_transform)

# check model accuracy score
print('RFC model accuracy score with '+str(n_estimators)+' decision trees:{0:0.4f}'. format(accuracy_score(labels_test, y_rfc)))

cmatrix_rfc = confusion_matrix(labels_test, y_rfc)

plt.figure()
with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':14,
                        'ytick.labelsize':14}):
    ax = sns.heatmap(cmatrix_rfc/np.sum(cmatrix_rfc), annot=True, fmt='.1%',cmap='Blues', annot_kws={'size':14})
plt.show()

#sns.heatmap(cmatrix_rfc/np.sum(cmatrix_rfc), annot=True, 
#            fmt='.2%', cmap='Blues')
#plt.show()

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['healthy and 5 mm', '10mm and added mass', '15 and 20mm'])
#cm_display.plot()
#plt.show()


print('--------------------------------------------------------------')
print('Classification details random forest classifier with '+str(n_estimators)+' trees')
print('Confusion matrix\n\n\ ',cmatrix_rfc)
print(classification_report(labels_test,y_rfc))
d=2