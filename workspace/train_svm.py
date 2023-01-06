from doctest import master
import pandas as pd
import os
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cupy as xp
sys.path.insert(1, '../lib/svmgpu/')
from svm import SVM





if __name__== "__main__":

    #data=pd.read_csv("/idiap/temp/ibmahmoud/evolang/Meerkats_project/eGeMAPSv02.csv").drop('name',axis=1)
    #target=data['class']
    #features=data.drop('class',axis=1)
    df=pd.read_csv("./features_extraction/eGeMAPSv02_functionals.csv")
    target=df.iloc[:,len(df.columns)-1]
    features=df.iloc[:,1:len(df.columns)-1]

    
    train_size=int(len(features)*0.8)
    test_size=len(features)-train_size

    X_train, X_test, y_train, y_test= train_test_split(features.values,target.values,test_size=test_size,random_state=42)
    X_train=xp.asarray(X_train)
    X_test=xp.asarray(X_test)
    y_train=xp.asarray(y_train)
    y_test=xp.asarray(y_test)
    svm=SVM(kernel='linear',kernel_params={'sigma': 15 },classification_strategy='ovr',x=X_train,y=y_train,n_folds=3,use_optimal_lambda=True)
    svm.fit(X_train,y_train)
    prediction=svm.predict(X_test)
    matrix=confusion_matrix(xp.asnumpy(y_test),xp.asnumpy(prediction))
    print(matrix)
    missclass=svm.compute_misclassification_error(X_test,y_test)

    

    print(missclass)



