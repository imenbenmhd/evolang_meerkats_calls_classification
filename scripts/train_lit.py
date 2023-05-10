import argparse
import json
from tokenize import group
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
from src.utils import utils
from src.models.lit import Lit
import random

from src.models.linear_model import linearmodel
from src.models.hannahcnn import HannahCNN
from src.models.cnn_16khz_seg import CNN_16KHz_Seg
from src.models.cnn_16khz_subseg import CNN_16KHz_Subseg
from src.models.Palazcnn import PalazCNN


from src.utils.logmelfilterbankspectrum import LogMelFilterBankSpectrum
from src.utils.featuresExtractor import featuresextraction

from src.data.nccrmeerkatsdataset import NCCRMeerkatsDataset
from src.data.mfcc import mfccMeerkatsDataset
from src.data.ut3dogsdataset import UT3dogsdataset
from src.data.acousticeventsdataset import AEDataset
from src.data.isabelmeerkatdataset import isabelMerkatDataset


from transformers import Wav2Vec2Model
import multiprocessing as mp

import sys
sys.path.append('../')

from meerkats import config
import warnings
warnings.filterwarnings("ignore")
# Wanb

# Map
with open('/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/workspace/src/data/class_to_index_isabel.json') as f:
    class_to_index = json.load(f)


def arg_parser():
        parser=argparse.ArgumentParser()
        group=parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
                '-dir',
                '--input_dir',
                help='the directory of the files to classify')
        parser.add_argument(
        '-s',
        '--sampling_rate',type=int,
        help='sampling rate to use')
        parser.add_argument(
        '-b',
        '--batch_size',type=int,
        help='batch size to use')
        parser.add_argument(
        '-f',
        '--framing', action='store_true', default=False,
        help='To frame the segment into 100ms frames or not')


        parser.add_argument('-lr','--learning_rate',type=float,help="the learning rate to use",default=None)
        parser.add_argument('-m','--model',default=None)
        args=parser.parse_args()
        return args

#wandb_logger = WandbLogger(name="test_1_RMS",project="Isabel_meerkat")
EPOCHS = 100
kfold=True

s3prl_dimensions={"wavlm" : 768, "hubert" : 768, "apc" : 512, "mockingjay" : 768, "npc" : 512, "wav2vec2": 768}

learning_rates=[0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.001,0.0015,0.003,0.005]

random.seed(42)
if __name__ == "__main__":
        
    # Data
    args=arg_parser()
    print(config.GITROOT)
    if args.model == None:
         transform=None
    else:
        transform=featuresextraction(upstream=args.model, layer=5)
        
    dataset_test=isabelMerkatDataset(
             audio_dir=args.input_dir,
             class_to_index=class_to_index,
             target_sample_rate=args.sampling_rate,
             train=False)
    dataset_train=isabelMerkatDataset(
            audio_dir=args.input_dir,
            class_to_index=class_to_index,
            target_sample_rate=args.sampling_rate,
            train=True)
    
    
    #data_train=isabelMerkatDataset(audio_dir=args.input_dir,class_to_index=class_to_index,target_sample_rate=args.sampling_rate,train=True,transform=transform)
    #,  transform= LogMelFilterBankSpectrum(frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))
    import ipdb; ipdb.set_trace();
    num_classes = len(set(class_to_index.values()))
    
    
    #effectif=(train_list.class_index.value_counts()).sort_index()
    #weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
    # Split Val and test
    
    if kfold==False:
        val_size = int(0.4* len(data_test))
        test_size = len(data_test) - val_size
        seed = torch.Generator().manual_seed(42)
        val_dataset, test_dataset = torch.utils.data.random_split(data_test, [val_size, test_size], generator=seed)

        print(f'There are {len(data_test)} data points in the test set and {num_classes} classes.')
        print(f'There are {len(data_train)} data points in the train set and {num_classes} classes.')
        train_loader = DataLoader(data_train, batch_size=args.batch_size,num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
        #effectif=(filelist.class_index.value_counts()).sort_index()
        #weights=torch.tensor(max(#effectif) / effectif, dtype=torch.float32)
        es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)
        trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=25, enable_progress_bar=True,callbacks=[es])
        #model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),args.learning_rate,framing=args.framing)
        model = Lit(linearmodel(728,num_classes),args.learning_rate,framing=args.framing,pretrained_model=None) # if using mfcc to classify

        trainer.tune(model,train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

        trainer.test(model, ckpt_path='best', dataloaders=test_loader)

    else:
    # k-folds
        result={}
        k_folds=5
        dataset=torch.utils.data.ConcatDataset([dataset_test,dataset_train])
        labels=dataset.datasets[0].filelist.class_index.tolist()+ dataset.datasets[1].filelist.class_index.tolist()
        #kfold=StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
        kfold=KFold(n_splits=5)
        if args.learning_rate is not None:
             learning_rates=[args.learning_rate]
        for lr in learning_rates:
                accuracies_folds=[]

                for fold,(train_ids,test_ids) in enumerate(kfold.split(dataset)):
                
                        print(f'Fold {fold}')
                        #x=pd.concat((dataset.datasets[0].filelist,dataset.datasets[1].filelist))
                        num_train = len(train_ids)
                        #indices = list(range(num_train))
                        split = int(numpy.floor(0.2* num_train))
                        #numpy.random.shuffle(indices)
                        train_idx, valid_idx = train_ids[split:], train_ids[:split]

                        # mask=dataset.filelist.index.isin(train_ids)
                        # dataset_=dataset.filelist[mask] 
                        # all_ids=numpy.arange(0,len(dataset))
                        # mask=x.index.isin(train_ids)
                        # dataset_=x[mask]
                        # for k, (train_idx, valid_idx) in enumerate(kfold.split(dataset_,dataset_.class_index)):
                        #         train_ids=train_idx
                        #         break
                        # all_ids=numpy.arange(0,len(dataset))
                        # keep_mask=numpy.isin(all_ids,test_ids,invert=True)
                        # new_ids=all_ids[keep_mask]
                        # train_ids=new_ids[train_ids]
                        # valid_idx=new_ids[valid_idx]
                        print(f'There are {len(test_ids)} data points in the test set and {num_classes} classes.')
                        print(f'There are {len(train_idx)} data points in the train set and {num_classes} classes.')
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
                        #pretrained=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
                        if transform is not None:
                                model = Lit(linearmodel(s3prl_dimensions[args.model],num_classes),lr,num_classes=num_classes,framing=args.framing,pretrained_model=None)
                        else:
                                model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),lr,num_classes=num_classes,framing=args.framing)

                        trainer.tune(model,train_dataloaders=train_loader)
                        trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
                        trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

                        accur=trainer.test(model, ckpt_path='best', dataloaders=test_loader)
                        accuracies_folds.append(accur[-1]["unbalanced accuracy"])
                result[lr]=accuracies_folds
   
        

        with open('test_isabelRMS.pkl', 'wb') as f:
                pickle.dump(result, f)
        torch.cuda.empty_cache()
        
