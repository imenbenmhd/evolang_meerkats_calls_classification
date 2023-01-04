from asyncio.unix_events import BaseChildWatcher
from traceback import print_tb
from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix
from torchmetrics import Recall
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from src.utils import utils
from src.models.hannahcnn import HannahCNN
from src.models.cnn_16khz_subseg import CNN_16KHz_Subseg
from src.models.Palazcnn import PalazCNN
import pandas as pd
import json 
import os
class Lit(pl.LightningModule):
    def __init__(self,model,learning_rate,fold=None,weight=None,framing=False):
        super().__init__()

       # self.model = PalazCNN(n_input=1,n_output=9,flatten_size=1)
        self.model= model
        self.probabilities=pd.DataFrame(columns=['probabilities','true'])
        self.fold=fold
        self.weight=weight
        self.frame=framing
        self.learning_rate=learning_rate
        self.save_hyperparameters()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        #optimizer=torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def common_step(self,batch,batch_idx):
        x,y=batch
        if self.frame:
            x=utils.frame_batch(x,100,5,16000)
            logits = self.model(x)
            logits=logits.sum(axis=0).unsqueeze(0)

            return logits,y
        else:
            preds=self.model(x)
            return preds,y
   
   


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        preds,y=self.common_step(batch,batch_idx)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loss = nn.CrossEntropyLoss(weight=self.weight.to(device))
        train_loss=loss(preds,y)
        train_acc = accuracy(preds, y)

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        preds,y=self.common_step(batch,batch_idx)
       
       
        loss = nn.CrossEntropyLoss()
        val_loss=loss(preds,y)
        val_acc = accuracy(preds, y)


        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        preds,y=self.common_step(batch,batch_idx)
        m=nn.Sigmoid()
        proba=m(preds)
        #new_row=pd.Series({"probabilities": proba.cpu().numpy(),"true": y.cpu().numpy()})
        #self.probabilities=pd.concat([self.probabilities,new_row.to_frame().T],ignore_index=True)
        #self.probabilities.to_pickle("/idiap/temp/ibmahmoud/evolang/proba.pkl")

        #logits=logits.sum(axis=0).unsqueeze(0)

        loss = nn.CrossEntropyLoss()
        test_loss=loss(preds,y)
        test_acc = accuracy(preds, y)

        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.fold !=None:
            self.probabilities.to_pickle("/idiap/temp/ibmahmoud/evolang/result_44k_100ms_5ms_weighted_newarch"+str(self.fold)+".pkl")
        return preds,y


    #def test_step_end(self, output_results):
        # this out is now the full size of the batch
      #  print(output_results)
        #all_test_labels=output_results.y
        #proba=nn.functional.softmax(all_test_step_outs,dim=1)

    def test_epoch_end(self, outputs): # outputs the preds and y of all the batchs.
    # do something with the outputs of all test batches
        conf_matrix,pred,targ=self.confusion_matrix(outputs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        acc=self.unbalanced_accuracy(conf_matrix)
        recal=Recall(num_classes=9).to(device)
        recall=recal(pred,targ).to(device)
        self.log("unbalanced accuracy", acc[0])
        self.log("Recall",recall)

        print(conf_matrix)

    def unbalanced_accuracy(self,matrix):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        accuracy=torch.diagonal(matrix) / torch.sum(matrix,dim=1)
        print(accuracy)
        accuracy=torch.sum(accuracy)/9
        return accuracy.to(device),

    def predict_step(self,batch,batch_idx):
        x,y=batch
        logits=self.model(x)
        proba=nn.functional.softmax(logits,dim=1)
        return proba,y

    def confusion_matrix(self,output): # output here is a list of preds and y with len=number of batches
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        predictions=torch.tensor(()).to(device)
        targets=torch.tensor(()).to(device)
        for i,x in output:
            pred=torch.argmax(i,dim=1)
            predictions= torch.cat((predictions, pred), 0)
            targets=torch.cat((targets,x),0)
        confmat = ConfusionMatrix(num_classes=9).to(device)
        targets=targets.type(torch.int64)
        predictions=predictions.type(torch.int64)
        matrix=confmat(predictions, targets)
          

        return matrix.to(device),predictions,targets


    
