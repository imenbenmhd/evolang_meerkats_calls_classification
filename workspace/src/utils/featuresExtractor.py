import argparse
import os

import numpy as np
import torch
import soundfile as sf

import sys

sys.path.insert(1,"/idiap/user/esarkar/speech/s3prl/")
print(os.getcwd())
from s3prl import hub
from s3prl.nn import S3PRLUpstream
from s3prl.util.download import set_dir





class featuresextraction(object):
    
    def __init__(self,upstream,layer):
        # print("init ok")
        self.model=upstream
        self.layer=layer
        
    def __call__(self,sample,device="cuda"):
        model=hub.wavlm()
        #model=S3PRLUpstream(self.model)
        #model.to(device)
        model.eval()
        sample = np.squeeze(sample.cpu().numpy())
        sample /=np.max(np.abs(sample))
        length=len(sample)

        sample=torch.from_numpy(sample).float()
        # hs,hs_len=model(sample.unsqueeze(0),torch.tensor(len(sample)).unsqueeze(0))
        hs=model(sample.unsqueeze(0))
        # for layer_id, (hs, hs_len) in enumerate(zip(hs, hs_len)):
            
        #     hs = hs.to("cpu")
        #     print(hs.shape)
        #     hs_len = hs_len.to("cpu")
        #     assert isinstance(hs, torch.FloatTensor)
        #     assert isinstance(hs_len, torch.LongTensor)
        #     print(layer_id)
        #     if layer_id == self.layer:
        #         hidden_states = hs
        #         hidden_states_len = hs_len

        # out_tensor=hidden_states[:,-1,:].clone().detach()
        out_tensor=hs["last_hidden_state"][:,-1,:]
        return out_tensor.clone().detach()
        
        
    

        
        
        
    
    
    