import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import sys
from torch.autograd import Variable


#faire cnn qui renvoit (B,T,d) ensuite faire torch.max qui renvoit sous la forme (B,d) ce qui ressemble au rnn


dn=50.

h=10
nepochs=100
dn=50.
kernel_size=3
stride=1
trainloss=[]
testloss=[]

#import data from meteo 2019
testing_data = pd.read_csv('2020.csv', sep=',',header=None)
training_data = pd.read_csv('2019.csv', sep=',', header=None)

#data description
#print(train_data.describe())
#print(train_data.head())

all_data_train = training_data[1].values.astype(float)
all_data_test = testing_data[1].values.astype(float)
#print(all_data)

trainx= all_data_train[:-7]/dn
#print(trainx.shape)
trainy= all_data_train[7:]/dn
testx= all_data_test[:-7]/dn
testy= all_data_test[7:]/dn



trainxn= torch.FloatTensor(trainx).view(1,-1,1)
trainyn= torch.FloatTensor(trainy).view(1,-1,1)
testxn= torch.FloatTensor(testx).view(1,-1,1)
testyn= torch.FloatTensor(testy).view(1,-1,1)

#print(trainxn.shape)
# print(trainyn.shape)
#print(testxn.shape)
# print(testyn.shape)



trainds = torch.utils.data.TensorDataset(trainxn, trainyn)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testxn, testyn)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()


class Net(nn.Module):
    def __init__(self, num_channels,nhid, kernel_size, stride):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)
        self.layerval=int(((358-(kernel_size-1))/stride))#formula in conv1d doc for output size
        #print(self.layerval)
        self.mlp = nn.Linear(self.layerval,7)


    def forward(self, x):
        xx=x.transpose(1,2)
        x = self.conv1(xx)
        #print(x.shape)
        x=torch.relu(x)
        T,B,H = x.shape
        #print("x.shape:",x.shape)
        #print(x.view(T*B,H).shape)
        x = self.mlp(x.view(T*B,H))
        return x

    
def test(mod):
    mod.train(False)
    loss_=[]
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        #print("inputs:",inputs.shape)
        #print("goldy:",goldy.shape)
        haty = mod(inputs)
        loss = crit(haty,goldy)
        totloss += loss.item()
        testloss.append(totloss)
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    for epoch in range(nepochs):
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty,goldy)
            totloss += loss.item()
            trainloss.append(totloss)
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        #print("err",totloss,testloss)
    #print("fin",totloss,testloss,file=sys.stderr)

model = Net(1, 1,kernel_size,stride)
train(model)
print(model(trainxn))

#loss plot
plt.plot(trainloss, label='train loss')
plt.plot(testloss, label='test loss')
plt.legend()
plt.show()

