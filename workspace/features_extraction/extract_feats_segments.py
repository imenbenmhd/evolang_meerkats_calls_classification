import numpy as np
import os
import pandas as pd
import soundfile as sf
import audiofile
import opensmile
import librosa
import json
if __name__=="__main__":
    csv_path_file="/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments/concat_path_class.csv"
    with open('/idiap/temp/ibmahmoud/evolang/evolang_meerkats_calls_classification/workspace/src/data/class_to_index.json') as f:
        class_to_index = json.load(f)
    wavlist=pd.DataFrame(librosa.util.find_files("/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments_/",ext=['wav']),columns=["path"])
    print(wavlist)
    wavlist['class_name']=wavlist.path.apply(lambda x: x.split('/')[-3])
    wavlist['class_index']=wavlist.class_name.apply(lambda x: class_to_index[x])
    labels=np.array([wavlist.class_index])
    
   
  

    
    smile=opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, feature_level=opensmile.FeatureLevel.Functionals)
    out_df=pd.DataFrame(columns=smile.feature_names)
    #out_df.insert(loc=0, column='A', value=wavfiles)
    for idx,finwav in enumerate(wavlist.path):
        signal,sampling=sf.read(finwav)
        features=smile.process_signal(signal,sampling)
        out_df = pd.concat([out_df,features],axis=0, ignore_index=True)
    out_df.insert(loc=0,column='path', value=wavlist.path)
    out_df.insert(loc=len(out_df.columns),column='label',value=wavlist.class_index)
    out_df.to_csv("/idiap/temp/ibmahmoud/evolang/Compare2016_functionals.csv",mode='w', header=None,index=False)