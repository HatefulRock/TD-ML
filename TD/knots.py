from logging import logProcesses
import torch
import numpy as np
import torch.nn as nn

global losses
losses = []

class Knots(nn.Module):

    def __init__(self):
        super(Knots, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        self.beta  = nn.Parameter(torch.Tensor([0.]))
        self.gamma = nn.Parameter(torch.Tensor([0.]))
        self.delta= nn.Parameter(torch.Tensor([0.]))


    def forward(self, X):
        return self.alpha + self.beta * X + self.gamma * X**2 
    

traind = torch.utils.data.TensorDataset(
        torch.Tensor([0,1,2,3,4,5])/10.,
        torch.Tensor([25,9,1,1,9,25])/100.)
train_loader = torch.utils.data.DataLoader(traind, batch_size=6, shuffle=True)

def learn():
    nepochs = 1000
    lossf = nn.MSELoss()
    mod = Knots()
    opt = torch.optim.Adam(mod.parameters(), lr=0.01)
    for ep in range(nepochs):
        n,totloss = 0,0.
        for x,y in train_loader:
            n+=1
            opt.zero_grad()
            haty = mod(x)
            loss = lossf(haty,y)
            totloss += loss.item()
            #print(f"loss: {loss.item()}")
            losses.append(loss.item())
            loss.backward()
            opt.step()
        totloss /= float(n)
        #print("TRAINLOSS",totloss,mod.alpha.item(),mod.beta.item())
    return mod

#plot training loss
import matplotlib.pyplot as plt
learn()
print(losses)
plt.plot(losses)
plt.show()

# #plot model
# mod = learn()
# x = np.linspace(0,0.5,100)
# y = mod(torch.Tensor(x)).detach().numpy()
# plt.plot(x,y)
# plt.show()







