from torch import nn

class MLP_3(nn.Module):
        def __init__(self, n_input, n_hidden, n_output):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output)
            )
            
        def forward(self, x):
            x = self.linear_relu_stack(x)
            return x
