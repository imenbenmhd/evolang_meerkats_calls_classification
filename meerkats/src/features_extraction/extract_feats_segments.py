from typing import Any
import numpy as np
import os
import pandas as pd
import soundfile as sf
import opensmile
import librosa
import json




class smile_features(object):

    def __init__(self, wavlist,name_feature):

        self.feature=name_feature
        self.wavlist=wavlist
        

    def extract(self):
        name=self.feature
        smile=opensmile.Smile(feature_set=opensmile.FeatureSet.,feature_level=opensmile.FeatureLevel.Functionals)
        features=np.empty()
        for idx,finwav in enumerate(self.wavlist):
            signal,fs=sf.read(finwav)
            features_=smile.process_signal(signal,fs)
            features= np.concat(features,features_,axis=0)
            print(features.shape)




