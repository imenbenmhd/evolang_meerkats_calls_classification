from itertools import permutations
import os
import csv
from pickletools import optimize
from turtle import backward
import pandas as pd
import torch 
import torch.optim as optim
import torch.nn as nn
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix
from src.models.linear_model import linearmodel
from sklearn.model_selection import train_test_split

import random
import numpy as np
from torchmetrics import ConfusionMatrix
import torchmetrics





if __name__ == "__main__": 
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df=pd.read_csv("/idiap/temp/ibmahmoud/evolang/evolang_meerkats_calls_classification/workspace/features_extraction/last_layer_features_nonoise.csv")
    target=df.iloc[:,len(df.columns)-1]
    features=df.iloc[:,:len(df.columns)-1]

    train_size=int(len(features)*0.8)
    test_size=len(features)-train_size

    X_train,X_test,y_train,y_test=train_test_split(features.values,target.values,test_size=test_size,random_state=42)
    feat_dim=X_train.shape[1]

    model=linearmodel(feat_dim, 9).to(dev)
    #model=PalazCNN().to(dev)

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-2)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion=nn.CrossEntropyLoss().to(dev)

    X_train=torch.from_numpy(X_train.astype(np.float32)).to(dev)
    y_train=torch.from_numpy(y_train).type(torch.LongTensor).to(dev)
    X_test=torch.from_numpy(X_test.astype(np.float32)).to(dev)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor).to(dev)
    bs=16
    model.train()
    for epoch in range(150):
        total_loss=0
        train_permutation=torch.randperm(len(X_train))
        for i in range(0,len(X_train),bs):
            indices=train_permutation[i:i+bs]
            inputs_x=X_train[indices].to(dev)
            y_pred=model(inputs_x)
            targets=y_train[indices]

            loss=criterion(y_pred.to(dev),targets)

            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'epoch', epoch,"total loss", total_loss)
        train_pred=model(X_train)
        accuracy=Accuracy().to(dev)
        total_acc=accuracy(train_pred,y_train)
        print(f'epoch', epoch,"total acc", total_acc)








            

    
    model.eval()
    test_loss = 0
    correct = 0
    X_test=X_test.unsqueeze(1)
    test_pred=model(X_test)
    test_pred=torch.argmax(test_pred,dim=1)
    test_pred=test_pred.type(torch.int64)

    confmat = ConfusionMatrix(num_classes=9).to(dev)
    matrix=confmat(test_pred, y_test)
    accuracy=torch.diagonal(matrix) / torch.sum(matrix,dim=1)
    print(accuracy)

    print(torch.sum(accuracy)/9)

    print(matrix)

