import argparse
import json
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger
from torchaudio import transforms
import numpy
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import KFold
import pickle

import sys

from src.utils import utils
from src.models.lit import Lit


import sys
sys.path.append('../')
from meerkats import config
from src.models.linear_model import linearmodel




from src.data.featuresSSLdataset import SSLdataset



# import warnings
# warnings.filterwarnings("ignore")
# Wanb

# Map
with open('src/data/class_to_index_marta.json') as f:
    class_to_index = json.load(f)


def arg_parser():
        parser=argparse.ArgumentParser()
        group=parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
                '-dir',
                '--input_dir',
                help='the directory of the pickles to classify')
        parser.add_argument(
        '-b',
        '--batch_size',type=int,
        help='batch size to use')

        parser.add_argument('-lr','--learning_rate',type=float,help="the learning rate to use",default=None)
        parser.add_argument('-m','--model',default=None)
        parser.add_argument('-d', '--device',default="cuda")
        args=parser.parse_args()
        return args

#wandb_logger = WandbLogger(name="test-frames",project="marta_meerkat")
EPOCHS = 100
kfold=True

s3prl_dimensions={"wavlm" : 768, "hubert" : 768, "apc" : 512, "mockingjay" : 768, "npc" : 512, "wav2vec2": 768}

learning_rates=[0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.001,0.0015,0.003,0.005]


if __name__ == "__main__":
    # Data
    args=arg_parser()
    import ipdb; ipdb.set_trace();

    dataset_test=SSLdataset(
            files_dir=args.input_dir,
            class_to_index=class_to_index,
            train=False)
    
    dataset_train=SSLdataset(
            files_dir=args.input_dir,
            class_to_index=class_to_index,
            train=True)
    
    

    num_classes = len(set(class_to_index.values()))
    


    # k-folds
    result={}
    k_folds=5
    dataset=torch.utils.data.ConcatDataset([dataset_test,dataset_train])
    labels=dataset.datasets[0].filelist.class_index.tolist()+ dataset.datasets[1].filelist.class_index.tolist()
    kfold=StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
    if args.learning_rate is not None:
            learning_rates=[args.learning_rate]
    for lr in learning_rates:
        accuracies_folds=[]

        for fold,(train_ids,test_ids) in enumerate(kfold.split(dataset,labels)):
                
            print(f'Fold {fold}')
            num_train = len(train_ids)
            split = int(numpy.floor(0.2* num_train))
            train_idx, valid_idx = train_ids[split:], train_ids[:split]

                        # mask=dataset.filelist.index.isin(train_ids)
                        # dataset_=dataset.filelist[mask] 
           
            print(f'There are {len(test_ids)} data points in the test set and {num_classes} classes.')
            print(f'There are {len(train_ids)} data points in the train set and {num_classes} classes.')
            print(f'There are {len(valid_idx)} data points in the validation set and {num_classes} classes.')
            train_subsampler=torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler=torch.utils.data.SubsetRandomSampler(valid_idx)
            test_subsampler=torch.utils.data.SubsetRandomSampler(test_ids)
                # Dataloader in this fold
            train_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=train_subsampler,collate_fn=utils.collate_fn, num_workers=4)
            val_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=val_subsampler,collate_fn=utils.collate_fn, num_workers=4)
            test_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=test_subsampler, shuffle=False,collate_fn=utils.collate_fn, num_workers=4)


                # Instantiate model
            es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)
            trainer = Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=75, enable_progress_bar=True)
            model = Lit(linearmodel(s3prl_dimensions[args.model],num_classes),lr,num_classes=num_classes)

            trainer.tune(model,train_dataloaders=train_loader)
            trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
            trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

            accur=trainer.test(model, ckpt_path='best', dataloaders=test_loader)
            accuracies_folds.append(accur[-1]["unbalanced accuracy"])
            result[lr]=accuracies_folds
    # with open("results_adam_last_layer.pkl","wb") as fp:
    #     pickle.dump(result,fp)
    # print("dict saved",result)
        

      
        
