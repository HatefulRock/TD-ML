import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math as m

loss_vals = []
train_losses = []

class Net(nn.Module):
    def __init__(self,nins,nout):
        super(Net, self).__init__()
        self.nins=nins
        self.nout=nout
        nhid = int((nins+nout)/2)
        self.hidden = nn.Linear(nins, nhid)
        self.out    = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x



def test(model, data, target):
    x=torch.FloatTensor(data)
    y=model(x).data.numpy()
    haty = np.argmax(y,axis=1)
    nok=sum([1 for i in range(len(target)) if target[i]==haty[i]])
    acc=float(nok)/float(len(target))
    return acc

def train(model, data, target):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x=torch.FloatTensor(data)
    y=torch.LongTensor(target)
    for epoch in range(100):
        #i=np.arange(len(x))
        #np.random.shuffle(i)
        epoch_loss=[]
        optim.zero_grad()
        haty = model(x)
        loss = criterion(haty,y)
        epoch_loss.append(loss.item())
        acc = test(model, data, target)
        print(str(loss.item())+" "+str(acc))
        loss.backward()
        optim.step()
        loss_vals.append(sum(epoch_loss)/len(x))
        #idx = np.arange(len(x))
    

# def genData(nins, nsamps):
#     prior0 = 0.7
#     mean0  = 0.3
#     var0   = 0.1
#     mean1  = 0.8
#     var1   = 0.01

#     n0 = int(nsamps*prior0)
#     x0=var0 * np.random.randn(n0,nins) + mean0
#     x1=var1 * np.random.randn(nsamps-n0,nins) + mean1
#     x = np.concatenate((x0,x1), axis=0)
#     y = np.ones((nsamps,),dtype='int64')
#     y[:n0] = 0
#     return x,y

def func(x,y):
    return m.exp(x+y)


def DataGenerator(number):
    data=[]
    x=np.linspace(-number,number,number)
    for j in range(len(x)):
        data.append([j,j,func(j,j)])
    return data


def toytest():
    model = Net(2,1)
    data=DataGenerator(20)
    inputdata=[[data[i][0],data[i][1]] for i in range(len(data))]
    target=[data[i][2] for i in range(len(data))]
    train(model,inputdata,target)


if __name__ == "__main__":
    toytest()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(loss_vals,label="val")

#plt.plot(train_losses,label="train")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
plt.legend()
plt.show()