import argparse
import json
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
from torchaudio import transforms
import numpy
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import KFold
import pickle

import sys



sys.path.append('../')
from meerkats import config
from meerkats.src.utils import utils
from meerkats.src.data.isabelmeerkatdataset import isabelMerkatDataset





from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer, TrainingArguments, Trainer


# import warnings
# warnings.filterwarnings("ignore")
# Wanb

import random
import argparse
import torch
import os
import sys

sys.path.insert(1,"/idiap/user/esarkar/speech/s3prl/")

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
        "-w", "--wavs", type=str, required=True, help="path to the input wav files"
    )

    parser.add_argument("--device", default="cuda", help="model.to(device)")
   

    parser.add_argument(
        "--b",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=16,
        help="sampling rate",
    )

    parser.add_argument("-lr", "--learning_rate", help="learning rate of model")
    return parser


def setup_directories(args):

    if args.expdir is None:
        args.expdir = f"result/downstream/{args.expname}"
    else:
        args.expdir = f"{args.expdir}/{args.expname}"

    os.makedirs(args.expdir, exist_ok=True)

    if args.cache_dir is not None:
        try:
            set_dir(args.cache_dir)
        except:
            print("Unable to set the path to -> {} \n".format(args.cache_dir))    




parser = get_argument_parser()
args = parser.parse_args()

with open(config.GITROOT + '/meerkats/src/data/class_to_index_marta1.json') as f:
    class_to_index = json.load(f)


class MeerkatCallsDataset(Dataset):
    def __init__(self, file_list, labels, tokenizer):
        self.file_list = file_list
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        audio_file = self.file_list[index]
        label = self.labels[index]
        return self.tokenizer(audio_file, return_tensors="pt", padding="longest"), label

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predicted_labels = torch.argmax(logits, axis=-1)
    accuracy = torch.sum(predicted_labels == labels).item() / len(labels)
    return {"accuracy": accuracy}




if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    device = args.device
    filelist = pd.read_csv(args.wavs)
    filelist['class_index'] = filelist.labels.apply(lambda x: class_to_index[x])



    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = MeerkatCallsDataset(filelist.path, filelist.class_index, tokenizer)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])

    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=9)
    training_args = TrainingArguments(
    output_dir="./output_dir",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)
    trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
