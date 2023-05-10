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

class CNN_16KHz_Subseg(nn.Module):
    def __init__(self, n_output,flatten_size=4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1,out_channels=128, kernel_size=30, stride=10)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128,out_channels=256, kernel_size=10, stride=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)

        self.conv3 = nn.Conv1d(in_channels=256,out_channels=512, kernel_size=4, stride=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=512,out_channels=512, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.adapt= nn.AdaptiveAvgPool1d(flatten_size)# this was added by Imen
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*flatten_size,out_features=512) # Imen changed from 10 to 512
        self.relu5 = nn.ReLU()
        self.fc = nn.Linear(512, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.adapt(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc(x)

        return x
