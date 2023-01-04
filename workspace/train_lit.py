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
import numpy
from sklearn.model_selection import KFold
from src.utils import utils
from src.models.lit import Lit
from src.models.hannahcnn import HannahCNN
from src.models.cnn_16khz_seg import CNN_16KHz_Seg
from src.models.cnn_16khz_subseg import CNN_16KHz_Subseg
from src.models.Palazcnn import PalazCNN
from src.data.nccrmeerkatsdataset import NCCRMeerkatsDataset


from transformers import Wav2Vec2FeatureExtractor

# Wanb

# Map
with open('src/data/class_to_index.json') as f:
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

wandb_logger = WandbLogger(name= "change_val_size",project="meerkats-subseg")

EPOCHS = 100
kfold=False

if __name__ == "__main__":
    # Data
    args=arg_parser()
    data_test = NCCRMeerkatsDataset(
            audio_dir=args.input_dir,
            class_to_index=class_to_index,
            target_sample_rate=args.sampling_rate,
            train=False,transform=False
            ) 
   
    
    data_train=NCCRMeerkatsDataset(audio_dir=args.input_dir,class_to_index=class_to_index,target_sample_rate=args.sampling_rate,train=True,transform=False)
    num_classes = len(set(class_to_index.values()))
    print(f'There are {len(data_test)} data points in the test set and {num_classes} classes.')
    print(f'There are {len(data_train)} data points in the train set and {num_classes} classes.')
    
   
   

    
    train_list=data_train._construct_filelist_dataframe()
    effectif=(train_list.class_index.value_counts()).sort_index()
    weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
    # Split
    val_size = int(0.4 * len(data_test))
    test_size = len(data_test) - val_size
    seed = torch.Generator().manual_seed(42)
    val_dataset, test_dataset = torch.utils.data.random_split(data_test, [val_size, test_size], generator=seed)
    if kfold==False:
        train_loader = DataLoader(data_train, batch_size=args.batch_size,collate_fn=utils.collate_fn,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,collate_fn=utils.collate_fn,num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn,num_workers=4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #effectif=(filelist.class_index.value_counts()).sort_index()
        #weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
        es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)

        trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=25, enable_progress_bar=True,callbacks=[es])
        
        model = Lit(PalazCNN(n_input=1,n_output=num_classes,flatten_size=1),args.learning_rate,framing=args.framing,weight=weights)

        trainer.tune(model,train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

        trainer.test(model, ckpt_path='best', dataloaders=test_loader)

        #pre = trainer.predict(dataloaders=test_loader)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #predictions=torch.tensor(()).to(device)
        #targets=torch.tensor(()).to(device)
        #for i,x in pre:
         #   pred=torch.argmax(i,dim=1)
          #  predictions= torch.cat((predictions, pred), 0)
          #  targets=torch.cat((targets,x),0)
        #confmat = ConfusionMatrix(num_classes=9).to(device)
        #targets=targets.type(torch.int64)
        #predictions=predictions.type(torch.int64)
        #matrix=confmat(predictions, targets)
          

        #trainer.test(model, ckpt_path='best', dataloaders=test_loader)

    else:
    # k-folds
        k_folds=5
        kfold=KFold(n_splits=k_folds,shuffle=True)
        dataset=ConcatDataset([data_test,val_dataset])
        for fold,(train_ids,val_ids) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold}')

            train_subsampler=torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler=torch.utils.data.SubsetRandomSampler(val_ids)
    # Dataloader in this fold
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,sampler=train_subsampler, collate_fn=utils.collate_fn)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,sampler=val_subsampler, collate_fn=utils.collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn)

    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate model
            model = Lit(HannahCNN(n_input=1, n_output=num_classes),fold=fold,weight=weights.to(device))
    # print(model)

    # Trains
            early_stop_callback=pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc",min_delta=0.00,patience=3,verbose=False,mode="max")
            trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=25, enable_progress_bar=True,callbacks=[early_stop_callback])
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Validate
            trainer.validate(model, ckpt_path='best', dataloaders=val_loader)

    # Test
            trainer.test(model, ckpt_path='best', dataloaders=test_loader)
