import torch
import torch.nn as nn

print("torch.cuda.is_available() == ", torch.cuda.is_available())


class smplNet(nn.Module):
    def __init__(self):
        super(smplNet, self).__init__()
        self.name = "smplNet"
        self.smplLinRelu = nn.Sequential(
                nn.Linear(36, 36),
                nn.ReLU(),
                nn.Linear(36, 18),
                nn.ReLU(),
                nn.Linear(18, 6)
                )

    def forward(self, x):
        output = self.smplLinRelu(x.to(torch.float32))
        return output
