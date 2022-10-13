import torch
import numpy as np
import torch.nn as nn


#Ex1
a=torch.ones([3,2])
c=torch.eye(2)
#print(a)
b=torch.sin(1+torch.sqrt(3*c)+5*torch.norm(a))
#print(b)

#Ex2
#implement a CNN with 3 layers 
#input tensor (t=10,d=1)
input=torch.randn(1,10)
#print(input)

class Net(nn.Module):
    def __init__(self,numChannels, classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=2, stride=1)
        


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
	
#model = Net(1, 10)
#print(model(input))

#Ex3

#initialize time series with 3000 samples
input=torch.randn(1,3000,1)
#print(input)

class Net(nn.Module):
    def __init__(self,numChannels, classes,nhid):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)
        self.rnn = nn.LSTM(1,10)
        self.mlp = nn.Linear(10,3)


    def forward(self, x):
        x = self.conv1(x.transpose(1,2))
        x=torch.relu(x)
        y,w=self.rnn(x.transpose(0,2).transpose(1,2))
        w=w[0]
        w=w.view(-1,10)
        w=self.mlp(w)
        return w

model = Net(10, 1,1000)
print(model(input))

