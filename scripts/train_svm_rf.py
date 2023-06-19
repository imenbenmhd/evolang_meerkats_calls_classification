from doctest import master
import argparse
import pandas as pd
import os
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
import json
import cupy as xp
import pickle
from sklearn import preprocessing

def get_argument_parser():

    parser = argparse.ArgumentParser(
        description="Extract the features from the pre-trained model"
    )

    parser.add_argument(
        "-m",
        "--model",
        help=
        "to chose between svm or random forest",
    )
    parser.add_argument(
    "-p",
     "--path",
     help=
     "the .csv file that containes the features and the label, created by extract_feats_segment"
    )
    parser.add_argument("-n","--name",help="the name of the features extracted")
    parser.add_argument("-d","--expdir",help="the directory to download the results")

    return parser







parser = get_argument_parser()
args = parser.parse_args()

def setup_directories(args):

    if args.expdir is None:
        args.expdir = f"result/{args.model}"
    else:
        args.expdir = f"{args.expdir}/{args.model}"

    os.makedirs(args.expdir, exist_ok=True)



def svm_class(args,X_train,y_train):
    uar_=make_scorer(uar,greater_is_better=True)
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid'],'decision_function_shape': ['ovo']}
    X_train = preprocessing.scale(X_train)
    grid=GridSearchCV(svm.SVC(),param_grid,scoring=uar_,refit=True,cv=5,verbose=2)
    grid.fit(X_train,y_train)

    return grid

def rf_class(args,X_train,y_train):

    param_grid = {
    "n_estimators": [10,20,30,40,50, 70,80, 100, 150, 200],
    "max_depth": [None, 5, 7 , 10],
    "min_samples_split": [2, 3,5, 7,10],
    "min_samples_leaf": [1, 2, 3,4]
}

      
    uar_=make_scorer(uar,greater_is_better=True)

    rf=RandomForestClassifier(random_state=42)
    grid= GridSearchCV(estimator=rf, param_grid=param_grid,scoring=uar_, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid


    
    

def uar(X,Y):

    confusion_m=metrics.confusion_matrix(X,Y)

    av_uar=np.sum(np.diag(confusion_m) / np.sum(confusion_m,axis=1))/confusion_m.shape[0]

    return av_uar



if __name__=="__main__":
    features=pd.read_csv(args.path,header=None,index_col=0)

    target=features.iloc[:,-1]
    features=features.iloc[:,:-1]

    X_train, X_test, y_train, y_test= train_test_split(features.values,target.values,test_size=0.2,random_state=42)

    if args.model=="svm":
        grid=svm_class(args,X_train,y_train)
        X_test= preprocessing.scale(X_test)

        y_pred=grid.predict(X_test)
    else:
        grid=rf_class(args,X_train,y_train)


        
        y_pred=grid.predict(X_test)

    confusion_m=metrics.confusion_matrix(y_test,y_pred)
    uar= np.diag(confusion_m) / np.sum(confusion_m,axis=1)

    av_uar=np.sum(np.diag(confusion_m) / np.sum(confusion_m,axis=1))/confusion_m.shape[0]
    result={"confusion matrix": confusion_m, "uar": uar, "av_uar": av_uar, "param": grid.best_params_}
    with open(args.model + args.name + "result_mara.pkl", "wb") as handle:
        pickle.dump(result, handle)











    



    





