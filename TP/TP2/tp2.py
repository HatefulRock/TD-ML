import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import sys
from torch.autograd import Variable


h=10
nepochs=100
dn=50.

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
trainy= all_data_train[7:]/dn
testx= all_data_test[:-7]/dn
testy= all_data_test[7:]/dn



trainxn= torch.FloatTensor(trainx).view(1,-1,1)
trainyn= torch.FloatTensor(trainy).view(1,-1,1)
testxn= torch.FloatTensor(testx).view(1,-1,1)
testyn= torch.FloatTensor(testy).view(1,-1,1)


# # # trainx = 1, seqlen, 1
# # # trainy = 1, seqlen, 1
trainds = torch.utils.data.TensorDataset(trainxn, trainyn)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testxn, testyn)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()


class Mod(nn.Module):
    def __init__(self,nhid):
        super(Mod, self).__init__()
        self.rnn = nn.RNN(1,nhid)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.rnn(xx)
        T,B,H = y.shape
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y

def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty,goldy)
        totloss += loss.item()
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
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        #print("err",totloss,testloss)
    print("fin",totloss,testloss,file=sys.stderr)

mod=Mod(h)
#print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod)

mod.eval()
dataX = Variable(torch.Tensor(np.array(trainx)))
dataY = trainy
data_predict = mod(trainxn).squeeze(0).data.numpy()*dn
dataY_plot = dataY

#data_predict = data_predict
#print(data_predict)
dataY_plot = dataY_plot*dn
#print(dataY_plot)
#length of trainx


plt.axvline(x=len(trainx), c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()
