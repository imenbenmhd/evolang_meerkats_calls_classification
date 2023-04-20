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
    def __init__(self,model,learning_rate,num_classes,fold=None,weight=None,framing=False,pretrained_model=None):
        super().__init__()

       # self.model = PalazCNN(n_input=1,n_output=9,flatten_size=1)
        self.model= model
        self.probabilities=pd.DataFrame(columns=['probabilities','true'])
        self.fold=fold
        self.weight=weight
        self.frame=framing
        self.learning_rate=learning_rate
        self.num_classes=num_classes
        self.pretrained=pretrained_model
        self.save_hyperparameters(ignore=['model'])



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
            if self.pretrained is not None:

                x=torch.squeeze(x,1)
                self.pretrained.eval()
                x=self.pretrained(x).last_hidden_state
                batch_size, sequence_length, hidden_size = x.shape
                padded_wav2vec2_output = torch.zeros((batch_size, 35, hidden_size)).to(x.device)
                padded_wav2vec2_output[:, :sequence_length, :] = x
                wav2vec2_output = padded_wav2vec2_output.view(batch_size, -1)
                #x.requires_drad=False
                #x=x[:,self.layer, :]

                #x=x.view(x.shape[0],-1)
                #print(x.size())
                #x=torch.unsqueeze(x,1)
            #print(x.mean(dim=1))
            preds=self.model(x)
            return preds,y
   
   


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        preds,y=self.common_step(batch,batch_idx)

        loss = nn.CrossEntropyLoss() #weight=self.weight.to(device))
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

        return val_acc

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
        preds_ = torch.argmax(preds, dim=1)

        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.fold !=None:
            self.probabilities.to_pickle("/idiap/temp/ibmahmoud/evolang/result_44k_100ms_5ms_weighted_newarch"+str(self.fold)+".pkl")
        cm = self.calculate_confusion_matrix(y, preds_)

        return {"confusion_matrix": cm},preds,y



    #def test_step_end(self, output_results):
        # this out is now the full size of the batch
      #  print(output_results)
        #all_test_labels=output_results.y
        #proba=nn.functional.softmax(all_test_step_outs,dim=1)
    def calculate_confusion_matrix(self,y_true, y_pred):

        confmat=ConfusionMatrix(num_classes=self.num_classes).to('cuda')
        cm = confmat(y_pred, y_true)
        return cm

    def test_epoch_end(self, outputs): # outputs the preds and y of all the batchs.
    # do something with the outputs of all test batches

        cm = sum([x["confusion_matrix"] for x,_,_ in outputs])
        acc=self.unbalanced_accuracy(cm)
        self.log("unbalanced accuracy", acc.item())

        print(acc)
        print(cm)
        return acc.item()
        #numpy.savetxt("confusion_matrix.txt", cm, fmt="%d")

        


    def unbalanced_accuracy(self,matrix):
        accuracy=torch.diagonal(matrix) / torch.sum(matrix,dim=1)
        print(accuracy)
        accuracy=torch.sum(accuracy)/self.num_classes
        return accuracy.to('cuda')

    def predict_step(self,batch,batch_idx):
        x,y=batch
        logits=self.model(x)
        proba=nn.functional.softmax(logits,dim=1)
        return proba,y

        
          



    
