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
with open('src/data/class_to_index_marta.json') as f:
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

#wandb_logger = WandbLogger(name="ADAM-pretrained-mara-on-marta-without-fixingweights",project="marta_meerkat")
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
    dataset =NCCRMeerkatsDataset(
            audio_dir=args.input_dir,
            class_to_index=class_to_index,
            target_sample_rate=args.sampling_rate,
            train=False,transform=transform)
    
    #data_train=isabelMerkatDataset(audio_dir=args.input_dir,class_to_index=class_to_index,target_sample_rate=args.sampling_rate,train=True,transform=transform)
    #,  transform= LogMelFilterBankSpectrum(frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))

    num_classes = len(set(class_to_index.values()))
    
    
    #effectif=(train_list.class_index.value_counts()).sort_index()
    #weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32)
    # Split Val and test
   

    # k-folds
    result={}
    #learning_rates=numpy.arange(0.0001,0.0015,0.0001)
    learning_rates=[0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.001,0.0015,0.003,0.005]
    k_folds=5
    kfold=StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
   
        #dataset=ConcatDataset([data_train,data_test])
    for lr in learning_rates:
        accuracies_folds=[]
        accuracies_folds_marta=[]
        for fold,(train_ids,test_ids) in enumerate(kfold.split(dataset,dataset.filelist.class_index)):
            import ipdb; ipdb.set_trace();
            print(f'Fold {fold}')

            #     num_train = len(train_ids)
            #     indices = list(range(num_train))
            #     split = int(numpy.floor(0.2* num_train))
            #     numpy.random.shuffle(indices)
            #     train_idx, valid_idx = indices[split:], indices[:split]
            mask=dataset.filelist.index.isin(train_ids)
            dataset_=dataset.filelist[mask] 
            all_ids=numpy.arange(0,len(dataset))
            for k, (train_idx, valid_idx) in enumerate(kfold.split(dataset_,dataset_.class_index)):
                 train_ids=train_idx
                 break
            all_ids=numpy.arange(0,len(dataset))
            keep_mask=numpy.isin(all_ids,test_ids,invert=True)
            new_ids=all_ids[keep_mask]
            train_ids=new_ids[train_ids]
            valid_idx=new_ids[valid_idx]
            print(f'There are {len(test_ids)} data points in the test set and {num_classes} classes.')
            print(f'There are {len(train_ids)} data points in the train set and {num_classes} classes.')
            print(f'There are {len(valid_idx)} data points in the validation set and {num_classes} classes.')

            train_subsampler=torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler=torch.utils.data.SubsetRandomSampler(valid_idx)
            test_subsampler=torch.utils.data.SubsetRandomSampler(test_ids)

        # Dataloader in this fold
            train_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=train_subsampler,collate_fn=utils.collate_fn, num_workers=8)
            val_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=val_subsampler,collate_fn=utils.collate_fn, num_workers=8)
            test_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=test_subsampler, shuffle=False,collate_fn=utils.collate_fn, num_workers=8)


        # Instantiate model
            es =pl.callbacks.early_stopping.EarlyStopping(monitor="train_acc", mode="max", patience=20)
            trainer = Trainer(logger=wandb_logger,accelerator='gpu', devices=1, max_epochs=EPOCHS, log_every_n_steps=75, enable_progress_bar=True)
                #pretrained=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            model = Lit(PalazCNN(n_output=7),learning_rate=1e-3,num_classes=7)
            modelm=Lit(PalazCNN(n_output=9),learning_rate=1e-3,num_classes=9)
            new_model=model.load_from_checkpoint("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/workspace/mara_meerkat/f3wfnwvj/checkpoints/epoch=99-step=25800-v1.ckpt") # path of kernel of 40 samples
            model_marta=modelm.load_from_checkpoint("/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification/workspace/marta_meerkat/3miufkp5/checkpoints/epoch=99-step=14400.ckpt")
            # for param in new_model.model.parameters():
            #     param.requires_grad = False
            new_model.model.fc1=nn.Linear(80,9)
            new_model.num_classes=9
            new_model.learning_rate=lr
            trainer.tune(new_model,train_dataloaders=train_loader)
            trainer.fit(new_model, train_dataloaders=train_loader,val_dataloaders=val_loader)
            trainer.validate(new_model, ckpt_path='best', dataloaders=val_loader)

            acc=trainer.test(new_model, ckpt_path='best', dataloaders=test_loader)
            accuracies_folds.append(acc[-1]["unbalanced accuracy"])
            print("Now best trained marta model")
            acc_marta=trainer.test(model_marta,dataloaders=test_loader)
            accuracies_folds_marta.append(acc_marta[-1]["unbalanced accuracy"])
    #if lr not in result:
        result[lr]={"mara" : accuracies_folds,"marta": accuracies_folds_marta}
    # with open("results_ADAM_sansfixingweight.pkl","wb") as fp:
    #     pickle.dump(result,fp)
    # print("dict saved",result)
        
            