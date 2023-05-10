from turtle import forward
import torch
import torch.nn as nn



class linearmodel(nn.Module):
    def __init__(self,feat_dim,n_output):
        super().__init__()
        self.layer=nn.Sequential(nn.Flatten(),nn.Linear(feat_dim,128))
        self.layer2=nn.Sequential(
                    nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),
                    nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,n_output))
    

    def forward(self,x):
        x=self.layer(x)
        x=self.layer2(x)

        return x

