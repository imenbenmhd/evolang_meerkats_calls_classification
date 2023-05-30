from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class PalazCNN(nn.Module):

    # Based on the paper: Towards End-to-end speech recognition
    # PDF: http://publications.idiap.ch/downloads/papers/2019/Muckenhirn_INTERSPEECH_2019.pdf
    
    def __init__(self, n_input=1, n_output=9, n_channel=40,flatten_size=1):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=40, stride=30)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=7, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        
        # Block 3
        self.conv3 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.adapt=nn.AdaptiveAvgPool1d(flatten_size) # arbitrary
        self.flatten=nn.Flatten()
        self.fc1   = nn.Linear(2*n_channel*flatten_size, n_output)
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x=self.adapt(x)
        

        x=self.flatten(x)
        intermediate=x
        x=self.fc1(x)
       
        return x
    
    
        

if __name__ == "__main__":
    model = PalazCNN()
    summary(model.cuda())
