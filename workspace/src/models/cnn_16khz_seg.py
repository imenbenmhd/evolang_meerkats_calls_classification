#!/usr/bin/python3

## Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
## Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>
## and Mathew Magimai Doss <mathew [at] idiap [dot] ch>. It was converted from the 
## original Keras-based framework to PyTorch by E. Sarkar <eklavya [dot] sarkar 
## [at] idiap [dot] ch>.
## 
## This file is part of RawSpeechClassification.
## 
## RawSpeechClassification is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License version 3 as
## published by the Free Software Foundation.
## 
## RawSpeechClassification is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with RawSpeechClassification. If not, see <http://www.gnu.org/licenses/>. 

from torch import nn
import torch.nn.functional as F

class CNN_16KHz_Seg(nn.Module):
    def __init__(self, n_input, n_output, n_channels=128, flatten_size=4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=n_channels, kernel_size=300, stride=100)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=2*n_channels, out_channels=2*2*n_channels, kernel_size=4, stride=2, padding=2)
        self.relu3 = nn.ReLU()

        self.conv1d4 = nn.Conv1d(in_channels=2*2*n_channels, out_channels=2*2*n_channels, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.adapt = nn.AdaptiveAvgPool1d(flatten_size)# <-- Arbitrary
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=flatten_size*2*2*n_channels, out_features=2*2*n_channels)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=2*2*n_channels, out_features=2*n_channels)
        self.relu6 = nn.ReLU()

        self.fc = nn.Linear(in_features=2*n_channels, out_features=n_output)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)             # [N, 128, 59]
        x = self.relu1(x)             # [N, 128, 59]
        x = self.maxpool1(x)          # [N, 128, 29]
        x = self.conv2(x)             # [N, 256, 13]
        x = self.relu2(x)             # [N, 256, 13]
        x = self.conv3(x)             # [N, 512, 5]
        x = self.relu3(x)             # [N, 512, 5]
        x = self.conv1d4(x)           # [N, 512, 3]
        x = self.relu4(x)             # [N, 512, 3]
        x = self.adapt(x)             # [N, 512, 4]
        x = self.flatten(x)           # [N, 2048]
        x = self.fc1(x)               # [N, 512]
        x = self.relu5(x)             # [N, 512]
        x = self.fc2(x)               # [N, 256]
        x = self.relu6(x)             # [N, 256]
        x = self.fc(x)                # [N, 9]
        return x
