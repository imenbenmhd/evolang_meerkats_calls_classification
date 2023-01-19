from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import random
import pandas as pd
import torchaudio
import itertools
import librosa
import torch
import json
import glob
import os

class mfccMeerkatsDataset(Dataset):
    def __init__(self,
                audio_dir=None,
                class_to_index=None,
                target_sample_rate=None,
                train=False
                ):
        
        self.audio_dir = audio_dir
        self.class_to_index=class_to_index
        self.target_sample_rate=target_sample_rate
        self.train=train

        self.filelist = self._construct_filelist_dataframe()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # Get variables
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)

        label = self._get_audio_sample_label(index, signal)

        # Preprocess (if necessary)
        signal = self._resample_if_necessary(signal, sr) # Downsample
        signal = self._mix_down_if_necessary(signal)     # Mono-channel

        # Return
        signal = librosa.feature.mfcc(y=signal.numpy(), sr=self.target_sample_rate,n_mfcc=40,n_fft=len(signal))
        signal=torch.tensor(signal)

        return signal, label

    def _construct_filelist_dataframe(self):
        
        
        # Read all file paths
        filelist = pd.DataFrame(librosa.util.find_files(self.audio_dir, ext=['wav']), columns=['path'])



       
       
        # Make dataframe
        filelist['segment_name'] = filelist.path.apply(lambda x: x.split('/')[-1])
        filelist['file_name'] = filelist.path.apply(lambda x: x.split('/')[-2])
        filelist['class_name'] = filelist.path.apply(lambda x: x.split('/')[-3])
        filelist['class_index'] = filelist.class_name.apply(lambda x: self.class_to_index[x])


        train_list,test_list=train_test_split(filelist,test_size=0.3,random_state=42)
        if self.train == False:
            return test_list
        if self.train: 
        
            
            #if self.transform:
             #   train_list=self._augmented_construct_filelist_dataframe(train_list)
            return train_list
        
        
    


    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        return self.filelist.path.iloc[index]

    def _get_audio_sample_label(self, index, signal):
        return self.filelist.class_index.iloc[index]









