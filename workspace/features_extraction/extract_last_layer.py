from doctest import OutputChecker
import torch
import numpy as np
import os
import json
from src.models.Palazcnn import PalazCNN
from src.models.lit import Lit
from src.data.nccrmeerkatsdataset import NCCRMeerkatsDataset
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from src.utils import utils


if __name__=="__main__":
    device=torch.device('cpu')
    path_="/idiap/temevolang/Meerkats_project/evolang.meerkats.call_type_classification/"
    lit=Lit(model=PalazCNN(n_input=1,n_output=9),learning_rate=1e-3)

   # model=lit.load_from_checkpoint(path_+"/meerkats-subseg/3oovaz0i/checkpoints/epoch=99-step=8400.ckpt") #best model
    model=lit.load_from_checkpoint(path_+"/meerkats-subseg/3sbjy52i/checkpoints/epoch=99-step=7900.ckpt") #best with data no noise
    model.eval()
    with open('src/data/class_to_index.json') as f:
        class_to_index = json.load(f)
        data_test = NCCRMeerkatsDataset(
            audio_dir="/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments_/",
            class_to_index=class_to_index,
            target_sample_rate=16000,
            train=False,transform=False
            ) 
   
    
    data_train=NCCRMeerkatsDataset(audio_dir="/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments_/",class_to_index=class_to_index,target_sample_rate=16000,train=True,transform=False)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    train_loader = DataLoader(data_train, batch_size=1,collate_fn=utils.collate_fn)
    FEATS=[] # list of all the features

# loop through batches of test then train
    for idx, inputs in enumerate(test_loader):
       
    # forward pass [with feature extraction]
        x,intermediate=model.model(inputs[0])
        output=np.c_[intermediate.detach().numpy(),inputs[1].detach().numpy()]

        FEATS.append(output)

    for idx, inputs in enumerate(train_loader):
        x,intermediate=model.model(inputs[0])
        output=np.c_[intermediate.detach().numpy(),inputs[1].detach().numpy()]
        FEATS.append(output)




    FEATS=np.array(FEATS)
    FEATS=FEATS.reshape((FEATS.shape[0]*FEATS.shape[1]), FEATS.shape[2]) # final features numpy

    df_out=pd.DataFrame(FEATS)
    df_out.to_csv(path_+"last_layer_features_nonoise.csv", mode="w", sep=",", header=None, index=False)
