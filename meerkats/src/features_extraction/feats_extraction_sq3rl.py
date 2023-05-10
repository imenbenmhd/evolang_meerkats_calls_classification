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

def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract the features from the pre-trained model"
    )
    upstreams = [attr for attr in dir(hub) if attr[0] != "_"]
    parser.add_argument(
        "-u",
        "--upstream",
        help=""
        'Upstreams with "_local" or "_url" postfix need local ckpt (-k) or config file (-g). '
        "Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. "
        "Please check upstream/README.md for details. "
        f"Available options in S3PRL: {upstreams}. ",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=-1,
        help=""
        "Layer number to extract features. "
        "0th layer corresponds to the first layer (cnn) "
        "The last layer is identified by (#layers - 1) due to zero indexing."
        "default: -1 (all layers)",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="List the number of layers in the model",
    )
    #parser.add_argument(
     #   "-w", "--wavs", type=str, required=True, help="path to the input wav files")
    
   # parser.add_argument(
       # "--ids", type=str, required=True, help="path to file with unique ids")
    parser.add_argument(
        "-n", "--expname", help="Save experiment at result/downstream/expname"
    )
    parser.add_argument("-p", "--expdir", help="Save experiment at expdir")
    parser.add_argument("--device", default="cuda", help="model.to(device)")
    parser.add_argument(
        "--cache_dir", help="The cache directory for pretrained model downloading"
    )
    parser.add_argument(
        "--npickle",
        type=int,
        default=-1,
        help="Number of utterances in a pickle file",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=5,
        help="Number of utterances in a batch. Should be a factor of npickle",
    )
    parser.add_argument("-f", "--picklename", help="Save the pickle with this basename")
    return parser

def get_model(upstream_model, device="cuda"):
    model= S3PRLUpstream(args.upstream)
    #model.to(device)
    model.eval()
    return model
parser = get_argument_parser()
args = parser.parse_args()

if __name__=="__main__":
    import ipdb; ipdb.set_trace();
    device=args.device
    layer_id_for_extraction=args.layer
    model=get_model(args.upstream,device)
    path="/idiap/project/evolang/meerkats_imen/dataset/isabel_data/isabel_data/processed_data/CutFilteredZeroCrossedCalls/220519_L_4_VLM273_08_CCSC_FRNT0_3_NN-VLF230_3M.wav"

    wav,sr=sf.read(path,dtype="float32")
    wav /=np.max(np.abs(wav))
    length=len(wav)

    wav=torch.from_numpy(wav).float()
    hs,hs_len=model(wav.unsqueeze(0),torch.tensor(len(wav)).unsqueeze(0).to(device))

    for layer_id, (hs, hs_len) in enumerate(zip(hs, hs_len)):
        hs = hs.to("cpu")
        hs_len = hs_len.to("cpu")
        assert isinstance(hs, torch.FloatTensor)
        assert isinstance(hs_len, torch.LongTensor)

        if layer_id == args.layer:
            hidden_states = hs
            hidden_states_len = hs_len


    


