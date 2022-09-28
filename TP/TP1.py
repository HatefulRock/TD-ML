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
        nhid = 5
        self.hidden = nn.Linear(nins, nhid)
        self.out    = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x



# def test(model, data, target):
#     x=torch.FloatTensor(data)
#     y=model(x).data.numpy()
#     haty = np.argmax(y,axis=0)
#     nok=sum([1 for i in range(len(target)) if target[i]==haty[i]])
#     acc=float(nok)/float(len(target))
#     return acc

def train(model, data, target):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    x=torch.FloatTensor(data)
    y=torch.FloatTensor(target)
    #print(x)
    #print(y)
    for epoch in range(500):
        epoch_loss=[]
        optim.zero_grad()
        haty = model(x)
        #print(haty)
        loss = criterion(haty,y)
        epoch_loss.append(loss.item())
        #acc = test(model, data, target)
        #print(str(loss.item())+" "+str(acc))
        loss.backward()

        optim.step()
        loss_vals.append(sum(epoch_loss)/len(x))
    

def func(x,y):
    return m.tan(x+y)


def DataGenerator(number):
    data=[]
    x=np.linspace(-number,number,100)
    for j in range(len(x)):
        data.append([j,j,func(j,j)])
    return data


def toytest():
    model = Net(2,1)
    data=DataGenerator(5)
    #inputdata=[[data[i][0],data[i][1]] for i in range(len(data))]

    #input create array of dimensions 20x2
    inputdata=np.array([[data[i][0],data[i][1]] for i in range(len(data))])
    #inputdata=[data[i][0] for i in range(len(data))]
    #print(inputdata)

    target=np.array([[data[i][2]] for i in range(len(data))])
    print(target)
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

