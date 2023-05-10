import argparse
import json
from tokenize import group
import pickle
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
from src.utils import utils
from src.models.lit import Lit

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

# Wanb

# Map
with open('src/data/class_to_index_isabel.json') as f:
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

wandb_logger = WandbLogger(name="ADAM-pretrained-mara-on-isabel-fixingweights",project="Isabel_meerkat")
EPOCHS = 100
kfold=True



if __name__ == "__main__":
    # Data
    args=arg_parser()
    FRAME_LENGTH = 25 * args.sampling_rate // 1000
    HOP_LENGTH = 10 * args.sampling_rate // 1000
    #transform = transforms.MFCC(sample_rate=args.sampling_rate, n_mfcc=40,melkwargs={"n_fft": 400, "hop_length": int(args.sampling_rate*0.002), "win_length": int(args.sampling_rate*0.005)})
    #transform=featuresextraction(upstream="wavlm", layer=5)
    transform=None
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

    num_classes = len(set(class_to_index.values()))
    
    
    #effectif=(train_list.class_index.value_counts()).sort_index()
    #weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
    # Split Val and test
   

    # k-folds
    result={}
    #learning_rates=numpy.arange(0.0001,0.0015,0.0001)
    if args.learning_rate is None:
        learning_rates=[0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.001,0.0015,0.003,0.005]
    else: 
         learning_rates=[args.learning_rate]
    k_folds=5
    dataset=torch.utils.data.ConcatDataset([dataset_test,dataset_train])
    labels=dataset.datasets[0].filelist.class_index.tolist()+ dataset.datasets[1].filelist.class_index.tolist()
    kfold=StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
    for lr in learning_rates:
        accuracies_folds=[]
        accuracies_folds_mara=[]
        for fold,(train_ids,test_ids) in enumerate(kfold.split(dataset,labels)):
            print(f'Fold {fold}')
            num_train=len(train_ids)
            split = int(numpy.floor(0.2* num_train))
            #     numpy.random.shuffle(indices)
            train_idx, valid_idx = train_ids[split:], train_ids[:split]
            
            print(f'There are {len(test_ids)} data points in the test set and {num_classes} classes.')
            print(f'There are {len(train_idx)} data points in the train set and {num_classes} classes.')
            print(f'There are {len(valid_idx)} data points in the validation set and {num_classes} classes.')

            train_subsampler=torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler=torch.utils.data.SubsetRandomSampler(valid_idx)
            test_subsampler=torch.utils.data.SubsetRandomSampler(test_ids)

        # Dataloader in this fold
            train_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=train_subsampler,collate_fn=utils.collate_fn, num_workers=8)
            val_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=val_subsampler,collate_fn=utils.collate_fn, num_workers=8)
            test_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=test_subsampler, shuffle=False,collate_fn=utils.collate_fn, num_workers=8)


        # Instantiate model
            checkpoint_marta=torch.load("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/experiments/experiment_1/epoch=99-step=7200-v1.ckpt") #best marta model
            checkpoint_mara=torch.load("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/experiments/experiment_2/epoch=99-step=12900.ckpt") #best mara model
            checkpoint_isa=torch.load("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/experiments/experiment_3/checkpoints/epoch=99-step=9600-v1.ckpt") #best isabel model
            model_marta = Lit(PalazCNN(n_output=9),learning_rate=0.003,num_classes=9)
            model_mara=Lit(PalazCNN(n_output=7),learning_rate=0.003,num_classes=7)
            model_isabel=Lit(PalazCNN(n_output=6),learning_rate=0.003,num_classes=6)
            model_marta.load_state_dict(checkpoint_marta["state_dict"])
            model_mara.load_state_dict(checkpoint_mara["state_dict"])
            model_isabel.load_state_dict(checkpoint_isa["state_dict"])


            es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)
            trainer = Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=75, enable_progress_bar=True)
            trainer_test= Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=75, enable_progress_bar=True)
                #pretrained=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            
            for param in model_mara.model.parameters():
                param.requires_grad = False
            model_mara.model.fc1=nn.Linear(80,6)
            model_mara.num_classes=6
            trainer.tune(model_mara,train_dataloaders=train_loader)
            trainer.fit(model_mara, train_dataloaders=train_loader,val_dataloaders=val_loader)
            trainer.validate(model_mara, ckpt_path='best', dataloaders=val_loader)

            acc=trainer.test(model_mara, ckpt_path='best', dataloaders=test_loader)
            accuracies_folds.append(acc[-1]["unbalanced accuracy"])
            print("Now best trained marta model")
            acc_mara=trainer_test.test(model_isabel,dataloaders=test_loader)
            accuracies_folds_mara.append(acc_mara[-1]["unbalanced accuracy"])
    #if lr not in result:
        result[lr]={"mara" : accuracies_folds,"isab": accuracies_folds_mara}
    with open("Mara_modelonisabel_fixdweights.pkl","wb") as fp:
        pickle.dump(result,fp)
    # print("dict saved",result)
        
            