from cgi import test
from importlib.resources import files
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

class UT3dogsdataset(Dataset):
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
        

        return signal, label

    def _construct_filelist_dataframe(self):
        
        
        # Read all file paths
        filelist = pd.DataFrame(librosa.util.find_files(self.audio_dir, ext=['wav']), columns=['path'])



       
       
        # Make dataframe
        filelist['file_name'] = filelist.path.apply(lambda x: x.split('/')[-1])
        filelist['class_name'] = filelist.path.apply(lambda x: x.split('/')[-2])
        filelist=filelist.loc[filelist["class_name"] !="S"]
        filelist['class_index'] = filelist.class_name.apply(lambda x: self.class_to_index[x])


        train_list,test_list=train_test_split(filelist,test_size=0.2,random_state=42)
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


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    #batch = [item.t() for item in batch] if batch is in 3D
    #batch= [item.permute(1,0,2) for item in batch]
    batch = [item.t() for item in batch]

    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # Iterate
    tensors, targets = [], []
    for waveform, label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets






if __name__ == "__main__":

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Variables
    SAMPLE_RATE = 44100

    # Class name -> Class index
    with open('class_to_index.json') as f:
        class_to_index = json.load(f)

    AUDIO_DIR = '/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments/'

    # Dataset
    data = NCCRMeerkatsDataset(
        audio_dir=AUDIO_DIR,
        class_to_index=class_to_index,
        target_sample_rate=SAMPLE_RATE,
        )

    # Dataloader
    dataloader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    print(f'There are {len(data)} samples in the dataset.')
