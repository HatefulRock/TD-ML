import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits import mplot3d

loss_tab = []
accuracy_tab=[]

class Net(nn.Module):
    def __init__(self,nins,nout):
        super(Net, self).__init__()
        self.nins=nins
        self.nout=nout
        nhid = 10
        self.hidden = nn.Linear(nins, nhid)
        self.out    = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x



def test(model, data, target):
    #implement mse accuracy formula
    x=torch.FloatTensor(data)
    y=model(x).detach().numpy()
    #print(y)
    ypred=torch.FloatTensor(target).detach().numpy()
    #print(ypred)
    acc=np.sqrt(np.mean((y-ypred)**2))
    
    return acc


def train(model, data, target):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    x=torch.FloatTensor(data)
    y=torch.FloatTensor(target)
    for epoch in range(500):
        optim.zero_grad()
        haty = model(x)
        loss = criterion(haty,y)
        loss_tab.append(loss.item())
        acc = test(model, data, target)
        accuracy_tab.append(acc)
        #print(str(loss.item())+" "+str(acc))
        loss.backward()
        optim.step()
    
    y=y.detach().numpy()
    haty=haty.detach().numpy()

    #draw 2d graph of network output
    plt.plot(y, label='y')
    plt.legend()
    plt.show()

    #draw plot of haty
    plt.plot(haty, label='haty')
    plt.legend()
    plt.show()

    

def func(x,y):
    return np.cos(x+y)


def DataGenerator(number_samples):
    #initialize training data with random values between -5 and 5
    train_data=np.random.uniform(-5,5,(number_samples,2))
    #print(train_data)
    test_data=func(train_data[:,0],train_data[:,1])
    test_data = np.expand_dims(test_data, axis=1)
    print(test_data)
    return train_data,test_data

def toytest():
    model = Net(2,1)
    training_data,training_target=DataGenerator(100)
    #inputdata=[[data[i][0],data[i][1]] for i in range(len(data))]

    #target=np.array([[data[i][2]] for i in range(len(data))])
    train(model,training_data,training_target)

    



if __name__ == "__main__":
    toytest()



plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(loss_tab,label="val")

#plot accuracy  
plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(accuracy_tab,label="val")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#plt.plot(train_losses,label="train")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
plt.legend()
plt.show()



