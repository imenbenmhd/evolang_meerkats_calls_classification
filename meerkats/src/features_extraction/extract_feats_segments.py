from typing import Any
import numpy as np
import os
import pandas as pd
import soundfile as sf
import opensmile
import librosa
import json

import torch
import sys
import pycatch22


from meerkats import config

from meerkats import src
from meerkats.src.models.Palazcnn import PalazCNN
from meerkats.src.models.lit import Lit
from meerkats.src.data.nccrmeerkatdataset import nccrMerkatDataset
from meerkats.src.utils import utils

from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="name of feature to extract"
    )
    parser.add_argument("-n","--name",help="name of the feature to extract")

    parser.add_argument(
        "-m",
        "--model",
        help=
        "if last layer path of the model of the best one", default=None
    )

    parser.add_argument( "-d","--data", help=" path of the .csv file for the data "

    )
    parser.add_argument("-i", "--index", help="path of the json dict or the config.dict")

    parser.add_argument(
    "-p",
     "--path",
     help=
     "the path to save the features "
    )
    parser.add_argument("-c","--classes",help="number of classe",default=None)
    parser.add_argument("-lr","--learning_rate",help="if last layer needed",default=None)
    return parser


parser = get_argument_parser()
args = parser.parse_args()


def setup_directories(args):

    if args.path is None:
        args.path= f"features/"
    else:
        args.path= f"{args.path}/"

    os.makedirs(args.path, exist_ok=True)





def get_dataset(args):

    with open(args.index) as f:
        class_to_index = json.load(f)
    path_data=config.DATADIR + args.data
    
    data_test = nccrMerkatDataset(
        audio_dir=path_data, class_to_index=class_to_index,
        target_sample_rate=16000,
        train=False) 
   
    
    data_train = nccrMerkatDataset(
        audio_dir=path_data,
        class_to_index=class_to_index,
        target_sample_rate=16000,
        train=True) 
    
    dataset=torch.utils.data.ConcatDataset([data_test,data_train])

    return dataset



def extract_embeddings(args,dataset):

        lit=Lit(model=PalazCNN(n_input=1,n_output=int(args.classes)),num_classes=int(args.classes),learning_rate=float(args.learning_rate))

        checkpoint=torch.load(args.model,map_location=torch.device('cpu')) #load the checkpoint given as argument
        lit.load_state_dict(checkpoint["state_dict"])
        lit.model.eval()

        loader = DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=utils.collate_fn, num_workers=4)

        FEATS=[] # list of all the features
        
        for batch in iter(loader):
            x,y=batch
            pred,intermediate=lit.model(x)
            output=np.c_[intermediate.detach().numpy(),y.detach().numpy()]
            FEATS.append(output)

        FEATS=np.array(FEATS)
        FEATS=FEATS.reshape((FEATS.shape[0]*FEATS.shape[1]), FEATS.shape[2]) # final features numpy
        df=pd.DataFrame(FEATS)

        return df
               
def extract_other_features(args,filelist):

    if args.name=="egemaps":
        smile=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)


    if args.name=="compare":
        smile=opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,feature_level=opensmile.FeatureLevel.Functionals)

    FEATS=[]

    for idx,finwav in enumerate(filelist.path):
        signal,fs=sf.read(finwav)
        audio_length=len(signal) / fs

        if audio_length < 0.1: # if length of file less than 100 ms
            repeat_times=int(0.1/audio_length) + 1
            signal=np.concatenate((np.array([signal]*repeat_times)),axis=0)
        features_=smile.process_signal(signal,fs)
        FEATS.append(features_.iloc[0].values)

    labels=filelist.class_index
    FEATS=np.array(FEATS)
    labels=np.array(labels)[:,None]

    FEATS=np.append(FEATS,labels,axis=1)
    df=pd.DataFrame(FEATS)
    return df



def extract_catch(dataset_list):
    FEATS=[]

    for idx,finwav in enumerate(dataset_list.path):
        signal,fs=sf.read(finwav)
        audio_length=len(signal) / fs
        if audio_length < 0.1:
            repeat_times=int(0.1/audio_length) + 1
            signal=np.concatenate((np.array([signal]*repeat_times)),axis=0)
        features=pycatch22.catch22_all(signal,catch24=True)
        FEATS.append(features["values"])
    
    labels=filelist.class_index
    FEATS=np.array(FEATS)
    labels=np.array(labels)[:,None]
    FEATS=np.append(FEATS,labels,axis=1)
    df=pd.DataFrame(FEATS)
    return df




if __name__== "__main__":
    setup_directories(args)

    dataset=get_dataset(args)
    filelist=pd.concat((dataset.datasets[0].filelist,dataset.datasets[1].filelist))

    if args.name == "embeddings":
        df=extract_embeddings(args,dataset)

    if args.name=="catch":
        df=extract_catch(filelist)


    else:
        df=extract_other_features(args,filelist)
    
    
    

    df.to_csv(args.path +args.name + "mara.csv",header=False)
        
    



