import torch
import torch.nn as nn

print("torch.cuda.is_available() == ", torch.cuda.is_available())


class smplNetCnt(nn.Module):
    def __init__(self):
        super(smplNetCnt, self).__init__()
        self.name = "smplNetCnt"
        self.smplLinRelu = nn.Sequential(
                nn.Linear(36, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                #nn.Sigmoid()
                nn.ReLU()
                )

    def forward(self, x):
        output = self.smplLinRelu(x.to(torch.float32))
        #print(" In Net: output = ", output)
        #output = torch.mul(output,10)
        #output = torch.argmax(output, dim=1).unsqueeze(1)
        #print(" In Net: After Mul: output = ", output)
        #output = torch.round(output)
        #print(" In Net: After Round: output = ", output)
        return output


class smplNetLog(nn.Module):
    def __init__(self):
        super(smplNetLog, self).__init__()
        self.name = "smplNetLog"
        self.smplLinRelu = nn.Sequential(
                nn.Linear(36, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
                #nn.Sigmoid()
                nn.Softmax()
                )

    def forward(self, x):
        output = self.smplLinRelu(x.to(torch.float32))
        #print(" In Net: output = ", output)
        #output = torch.mul(output,10)
        #output = torch.argmax(output, dim=1).unsqueeze(1)
        #print(" In Net: After Mul: output = ", output)
        #output = torch.round(output)
        #print(" In Net: After Round: output = ", output)
        return output
