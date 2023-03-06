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

class NCCRMeerkatsDataset(Dataset):
    def __init__(self,
                audio_dir=None,
                class_to_index=None,
                target_sample_rate=None,
                train=False,
                transform=None
                ):
        
        self.audio_dir = audio_dir
        self.class_to_index=class_to_index
        self.target_sample_rate=target_sample_rate
        self.train=train
        self.transform=transform

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
        signal = self._transform_if_necessary(signal)  # Transformation

        # Return
        


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
        
        
    def _augmented_construct_filelist_dataframe(self,filelist):

        tempdir="/idiap/temp/ibmahmoud/evolang/tmp/"

        augmented_list=filelist.copy()
        effectif=(filelist.class_index.value_counts()).sort_index()
        weights=torch.tensor(max(effectif) / effectif, dtype=torch.float32) # start by adding for the 2 smallest one
        values, index= torch.topk(weights,2)
        to_augment=filelist[filelist['class_index']==index[0]]
        for i in index[1:]:
            to_augment=pd.concat([to_augment,filelist[filelist['class_index']==i.item()]]) # these are the file that we will augment
        files_toaugment=to_augment['path'].tolist()
        speed_factor = [0.75,0.85,0.95, 0.8]
        for ind,file in enumerate(files_toaugment):
            for s in speed_factor:
                path_=change_speed_save(file,s,self.target_sample_rate,tempdir)
            
            
            
                row={'path': os.path.join(path_,file.split('/')[-1][:-4]+str(s)+".wav"), 'segment_name': file.split('/')[-1][:-4]+str(s)+".wav", 'file_name': file.split('/')[-2],'class_name': file.split('/')[-3],'class_index':self.class_to_index[file.split('/')[-3]]}
                new_df = pd.DataFrame([row])
                augmented_list = pd.concat([augmented_list, new_df], axis=0, ignore_index=True)
            #for d in db:   # adding a background noise decrease the validation accuracy too much
            #    path__=background_noise_save(d,file,tempdir)
             #   row={'path': os.path.join(path__,file.split('/')[-1][:-4]+str(d)+".wav"), 'segment_name': file.split('/')[-1][:-4]+str(d)+".wav", 'file_name': file.split('/')[-2],'class_name': file.split('/')[-3],'class_index':self.class_to_index[file.split('/')[-3]]}
              #  new_df = pd.DataFrame([row])
                #augmented_list = pd.concat([augmented_list, new_df], axis=0, ignore_index=True)


        return augmented_list


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
    
    def _transform_if_necessary(self, signal):

        if self.transform:
                signal = self.transform(signal)
                return signal


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    #batch = [item.t() for item in batch] if batch is in 3D
    #batch= [item.permute(1,0,2) for item in batch]
    batch = [item.permute(1,0,2) for item in batch]
    print(batch.size())
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1,3)


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


def change_speed_save(file,speed_factor,sample_rate=44100,dir=None):
    audio, sr= torchaudio.load(file)


    sox_effects = [
        ["speed", str(speed_factor)],
        ["rate", str(sample_rate)],
        ]
    transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, sr, sox_effects)
    new_path=os.path.join(dir,file.split('/')[-2])
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    torchaudio.save(os.path.join(new_path,file.split('/')[-1][:-4]+str(speed_factor)+".wav"), transformed_audio, sr)

    return new_path
 
def background_noise_save(db,file,dir=None):
    speech,sr= torchaudio.load(file)
    noise, _=torchaudio.load(os.path.join("/idiap/temp/ibmahmoud/evolang/tmp/noise","train.wav"))
    noise= noise[:,:speech.shape[1]]

    speech_rms= speech.norm(p=2)
    noise_rms=noise.norm(p=2)
    snr= 10** (db/20)
    scale= snr*noise_rms / speech_rms
    noisy_speech = (scale*speech +noise)/2
    new_path=os.path.join(dir,file.split('/')[-2])
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    torchaudio.save(os.path.join(new_path,file.split('/')[-1][:-4]+str(db)+".wav"), noisy_speech, sr)
    return new_path



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
