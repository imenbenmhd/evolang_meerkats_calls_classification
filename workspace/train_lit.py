import argparse
from email import parser
import json
from tokenize import group
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger
from torchaudio import transforms
import numpy
from sklearn.model_selection import KFold
from src.utils import utils
from src.models.lit import Lit
from src.models.hannahcnn import HannahCNN
from src.models.cnn_16khz_seg import CNN_16KHz_Seg
from src.models.cnn_16khz_subseg import CNN_16KHz_Subseg
from src.models.Palazcnn import PalazCNN

from src.data.nccrmeerkatsdataset import NCCRMeerkatsDataset
from src.data.mfcc import mfccMeerkatsDataset
from src.data.ut3dogsdataset import UT3dogsdataset
from src.data.acousticeventsdataset import AEDataset


#from transformers import Wav2Vec2FeatureExtractor

# Wanb

# Map
with open('src/data/class_to_index_UT3.json') as f:
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


        parser.add_argument('-lr','--learning_rate',type=float,help="the learning rate to use")
        args=parser.parse_args()
        return args

wandb_logger = WandbLogger(name= "UT3-kfoldtrue",project="dogs")

EPOCHS = 100
kfold=True

if __name__ == "__main__":
    # Data
    args=arg_parser()
    #transform = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=40,melkwargs={"n_fft": 400, "hop_length": int(args.sampling_rate*0.002), "win_length": int(args.sampling_rate*0.005)})
    data_test =UT3dogsdataset (
            audio_dir=args.input_dir,
            class_to_index=class_to_index,
            target_sample_rate=args.sampling_rate,
            train=False
            ) 
   
    
    data_train=UT3dogsdataset(audio_dir=args.input_dir,class_to_index=class_to_index,target_sample_rate=args.sampling_rate,train=True)

    num_classes = len(set(class_to_index.values()))
   
    
    
    #effectif=(train_list.class_index.value_counts()).sort_index()
    #weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
    # Split Val and test
    val_size = int(0.6* len(data_test))
    test_size = len(data_test) - val_size
    seed = torch.Generator().manual_seed(42)
    val_dataset, test_dataset = torch.utils.data.random_split(data_test, [val_size, test_size], generator=seed)
    if kfold==False:
        print(f'There are {len(data_test)} data points in the test set and {num_classes} classes.')
        print(f'There are {len(data_train)} data points in the train set and {num_classes} classes.')
        train_loader = DataLoader(data_train, batch_size=args.batch_size,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
        #effectif=(filelist.class_index.value_counts()).sort_index()
        #weights=torch.tensor(max(#effectif) / effectif, dtype=torch.float32)
        es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)

        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=25, enable_progress_bar=True,callbacks=[es])
        
        #model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),args.learning_rate,framing=args.framing)
        model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),args.learning_rate,framing=args.framing) # if using mfcc to classify

        trainer.tune(model,train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

        trainer.test(model, ckpt_path='best', dataloaders=test_loader)

    else:
    # k-folds
        k_folds=5
        kfold=KFold(n_splits=k_folds,shuffle=True)
        dataset=ConcatDataset([data_train,val_dataset])
        for fold,(train_ids,test_ids) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold}')
            num_train = len(train_ids)
            indices = list(range(num_train))
            split = int(numpy.floor(0.2* num_train))
            numpy.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
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

            trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=25, enable_progress_bar=True,callbacks=[es])
                
            model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),args.learning_rate,num_classes=num_classes,framing=args.framing)

            trainer.tune(model,train_dataloaders=train_loader)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

            trainer.test(model, ckpt_path='best', dataloaders=test_loader)
