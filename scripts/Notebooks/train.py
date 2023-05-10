from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import optim
from torch import nn
import torchaudio
import wandb
import torch
import argparse
import json

from src.data.nccrmeerkatsdataset import NCCRMeerkatsDataset
from src.models.hannahcnn import HannahCNN
from src.models.deepcnn import DeepCNN
from src.models.cnn_16khz_subseg import CNN_16KHz_Subseg
from src.models.cnn_16khz_seg import CNN_16KHz_Seg
from src.utils import utils

# Wanb
wandb.init(project="meerkats-cnn", entity="eklavya-team")

# Map: Class name -> Class index
with open('class_to_index.json') as f:
    class_to_index = json.load(f)

# Variables
AUDIO_DIR = "/idiap/temp/ibmahmoud/evolang/animal_data/Meerkat_sound_files_examples_segments/"
SAMPLE_RATE = 44100
BATCH_SIZE = 16
EPOCHS = 100

# def parse_arguments():
#     """Parse arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="CNN_16KHz_Subseg")
#     parser.add_argument("--arch", type=str, default="subseg")
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--random_seed", type=int, default=101)
#     parser.add_argument("--exp_dir", type=str)
#     parser.add_argument("--data_dir", type=str, default=)
#     parser.add_argument("--std_norm", action="store_true")
#     parser.add_argument("--instance_norm", action="store_true")
#     parser.add_argument("--use_balanced_idxs", action="store_true")
#     return parser.parse_args()

if __name__ == "__main__":

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Arguments
    # args = parse_arguments()
    
    # Dataset
    data = NCCRMeerkatsDataset(
        audio_dir=AUDIO_DIR,
        class_to_index=class_to_index,
        target_sample_rate=SAMPLE_RATE,
        )

    # Split train-test
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn)

    # Labels
    num_classes = len(set(class_to_index.values()))

    # Model
    model = HannahCNN(n_input=250, n_output=num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Send to device
    criterion.to(device)
    model.to(device)

    # Train
    utils.train_model(model, train_loader, criterion, optimizer, device, EPOCHS)

    # Save
    # torch.save(model.state_dict(), "cnnnetwork.pth")

    # Test
    utils.test_model(model, test_loader, device)