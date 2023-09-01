# Rocket impports 
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Classifiers imports 
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Classifications Accuracy imports
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

import numpy as np

def rocket_feature_extraction(train_rocket, test_rocket): 

    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.fit(train_rocket.numpy())
    X_train_transform = minirocket_multi.transform(train_rocket.numpy())
    X_test_transform = minirocket_multi.transform(test_rocket.numpy())

    return X_train_transform, X_test_transform


def ridge_classifier(train_features, y_train, test_features, y_test): 

    X_train_transform = train_features
    X_test_transform  = test_features


    ridge_clf = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17) ) 
    ridge_clf.fit(X_train_transform, y_train)
    y_pred = ridge_clf.predict(X_test_transform)

    ridge_acc = accuracy_score(y_test, y_pred)
    ridge_pre = precision_score(y_test, y_pred, average='weighted')
    ridge_c_matrix = confusion_matrix(y_test, y_pred)

    return ridge_c_matrix, ridge_acc, ridge_pre

def random_forest_classifier(train_features, y_train, test_features, y_test): 

    X_train_transform = train_features
    X_test_transform  = test_features

    rfc = RandomForestClassifier(n_estimators=800, max_features="sqrt")
    rfc.fit(X_train_transform, y_train)
    y_pred = rfc.predict(X_test_transform)

    rfc_acc = accuracy_score(y_test, y_pred)
    rfc_pre = precision_score(y_test, y_pred, average='weighted')
    rfc_c_matrix = confusion_matrix(y_test, y_pred)

    return rfc_c_matrix, rfc_acc, rfc_pre
