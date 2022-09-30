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
        nhid = 5
        self.hidden = nn.Linear(nins, nhid)
        self.out    = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x



# def test(model, data, target):
#     x=torch.FloatTensor(data)
#     y=model(x).detach().numpy()
#     haty = np.argmax(y,axis=1)
#     nok=sum([1 for i in range(len(target)) if target[i]==haty[i]])
#     acc=float(nok)/float(len(target))
#     return acc

def test(model, data, target):
    #implement mse accuracy formula
    x=torch.FloatTensor(data)
    y=model(x).detach().numpy()
    ypred=torch.FloatTensor(target).detach().numpy()

    acc=np.sqrt(np.mean((y-ypred)**2))
    return acc


# def test(model, data, target):
#     x=torch.FloatTensor(data)
#     y=np.squeeze(model(x).data.numpy())
#     return np.sqrt(np.mean((np.squeeze(target)-y)**2))



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
    plt.plot(haty, label='haty')
    plt.legend()
    plt.show()


    # #draw 3d graph of output of network
    # fig=plt.figure()
    # ax=plt.axes(projection='3d')
    # n_point=100
    # ax.plot(x[:n_point,0], x[:n_point,1], haty[:n_point], 'b', markersize=0.5)
    # plt.show()
    

def func(x,y):
    return np.cos(x+y)


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
    train(model,inputdata,target)

    # #draw 3d graph of output of network
    # x=np.linspace(-5,5,100)
    # y=np.linspace(-5,5,100)
    # Z=np.transpose(model(torch.FloatTensor(inputdata)).detach().numpy()).tolist()[0]
    # print(Z)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(x, y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    



if __name__ == "__main__":
    toytest()


# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)

# X, Y = np.meshgrid(x, y)
# Z = func(X, Y)
# print(X)

# print(Z)






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



