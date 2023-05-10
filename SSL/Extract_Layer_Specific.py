#!/usr/bin/env python
# coding: utf-8
# Author : Tilak 
# In[1]:

import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys


# In[7]:


def average_layers(features_dict, key, l_start, l_end):
    y =0
    for i in range (l_start, l_end):
        y = y + features_dict[key][i]
    return y/(l_end - l_start)

def load_features(features_file):
    with open(features_file, "rb") as handle:
        features_dict = pickle.load(handle)
    return features_dict

def dump_features(features_dict, features_file):
    with open(features_file, "wb") as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def sav_layer_specific_embd (features_dict, key, sav_dir):
    layer_0 = features_dict[key][0]  # the CNN output
    layer_1 = features_dict[key][1]  ## 1st transformer layer
    layer_mid = features_dict[key][6] ## the middle (6th) transformer layer
    layer_n = features_dict[key][-1]  ## the last (13th) transformer layer
    mean_layers = average_layers(features_dict,key, 1 ,13)  ## average from Transformer L1 to Transformer L13 (last layer).


    embd_dict = {
        "wav_nom":  key,  #normally it is key.split(" ")[0]
        #"lable" : key.split(" ")[1],
        "cnn_out" : layer_0,
        "tfmr_1" : layer_1,
        "tfmr_Mid": layer_mid,
        "tfmr_Last" : layer_n,
        "tfrm_mean" : mean_layers
    }
    file_nom = embd_dict["wav_nom"].split(".")[0].replace("/",".")
    feature_file = "{}/{}.pkl".format(sav_dir, file_nom)
    dump_features(embd_dict, feature_file)
    
def create_pkl_files(path, sav_path):
    features_dict = load_features(path)
    wav_key = list(features_dict.keys())
    #sav_path = "/idiap/temp/tpurohit/MoE/Iemocap_embd/S1/embd/wav2vec2L_embdFiles"
    for wavs in wav_key:
        sav_layer_specific_embd (features_dict, wavs, sav_path)



if __name__ == '__main__':
    import ipdb; ipdb.set_trace();
    main_fldr = sys.argv[1] # folder where s3prl embeddings (derived via s3prl wrapper) are saved 
    sav_path = sys.argv[2] # path where you want to save layer specific embeddings (extracted via this code)
    os.makedirs(sav_path, exist_ok=True)
    #main_fldr = "/idiap/temp/tpurohit/MoE/Iemocap_embd/S1/embd/wav2vec2L_Feats"
    all_embd_wav = glob(main_fldr+"/*.pkl", recursive = True)
    for embd_files in tqdm(all_embd_wav):
        create_pkl_files(embd_files, sav_path)
    

