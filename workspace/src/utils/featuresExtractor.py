import argparse
import os

import numpy as np
import torch
import soundfile as sf
import multiprocessing as mp

import sys

sys.path.insert(1,"/idiap/user/esarkar/speech/s3prl/")
from s3prl import hub
from s3prl.nn import S3PRLUpstream
from s3prl.util.download import set_dir
import s3prl





class featuresextraction(object):
    
    def __init__(self,upstream,layer):
        # print("init ok")
        self.model=upstream
        self.layer=layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    def __call__(self,sample):
        #model=hub.wavlm()
        model=S3PRLUpstream(self.model)
        #model=hub.load(self.model)
        #model.to(device)
        model.eval()
        #tokenizer = s3prl.models.__dict__[self.model + "_tokenizer"].from_pretrained(self.model)
        #model = s3prl.models.__dict__[self.model].from_pretrained(self.model)
        sample = sample.squeeze()
        sample /= torch.max(torch.abs(sample))
        length=len(sample)

        with torch.no_grad():

            hs,hs_len=model(sample.unsqueeze(0),torch.tensor(len(sample)).unsqueeze(0))
            #hs,hs_len=model(sample.unsqueeze(0),torch.tensor(len(sample)).unsqueeze(0))
            #features=model.extract_features(sample.unsqueeze(0), offset=0.0, duration=None, trim=False, use_grad=False, layer=-1)
        
        # for layer_id, (hs, hs_len) in enumerate(zip(hs, hs_len)):
            
        #     hs = hs.to("cpu")
        #     print(hs.sh
        #     hs_len = hs_len.to("cpu")
        #     assert isinstance(hs, torch.FloatTensor)
        #     assert isinstance(hs_len, torch.LongTensor)
        #     print(layer_id)
        #     if layer_id == self.layer:
        #         hidden_states = hs
        #         hidden_states_len = hs_len

        # out_tensor=hidden_states[:,-1,:].clone().detach()
        out_tensor=hs[-1]
        out_tensor=out_tensor[:,-1,:]
        out_tensor=out_tensor.mean(dim=0)
        return out_tensor.clone().detach()
        
        
    

        
        
        
    
    
    