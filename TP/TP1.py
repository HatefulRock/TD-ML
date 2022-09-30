import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math as m
import plotly.graph_objects as go


class Net(nn.Module):
    def __init__(self, nins, nout):
        super(Net, self).__init__()
        self.nins = nins
        self.nout = nout
        nhid = 10
        self.hidden = nn.Linear(nins, nhid)
        self.out = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x


def test(model, data, target):
    # implement mse accuracy formula
    x = torch.FloatTensor(data)
    y = model(x).detach().numpy()
    ypred = torch.FloatTensor(target).detach().numpy()

    mse = np.sqrt(np.mean((y-ypred)**2))
    return mse


def train(model, data, target, test_data, test_target):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_tab = []
    loss_test_tab = []

    x = torch.FloatTensor(data)
    y = torch.FloatTensor(target)
    for epoch in range(1000):
        optim.zero_grad()
        haty = model(x)
        loss = criterion(haty, y)
        loss_tab.append(loss.item())
        loss_test = test(model, test_data, test_target)
        loss_test_tab.append(loss_test)
        loss.backward()
        optim.step()
    xprime = x.detach().numpy()
    y = y.detach().numpy()
    haty = haty.detach().numpy()
    value_x = [elmt[0] for elmt in xprime]
    value_y = [elmt[1] for elmt in xprime]
    value_z = y.squeeze()
    computed_z = haty.squeeze()

    # draw 3d graph of network output

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(z=computed_z, x=value_x, y=value_y, mode='markers',
                               name='computed value'))
    fig.add_trace(go.Scatter3d(z=value_z, x=value_x, y=value_y, mode='markers',
                               name='real function value'))
    fig.show()

    # draw Mean square error metric for training and test data

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_tab, x=np.arange(len(loss_tab)),
                             mode='lines',
                             name='MSE on training data'))
    fig.add_trace(go.Scatter(y=loss_test_tab, x=np.arange(len(loss_test_tab)),
                             mode='lines',
                             name='MSE on test data'))
    fig.show()


def func(x, y):
    return np.cos(2*x+y)


def DataGenerator(number_samples):
    # initialize training data with random values between -5 and 5
    train_data_input = np.random.uniform(-5, 5, (int(0.8*number_samples), 2))
    train_data_output = func(train_data_input[:, 0], train_data_input[:, 1])
    train_data_output = np.expand_dims(train_data_output, axis=1)
    test_data_input = np.random.uniform(-5, 5, (int(0.2*number_samples), 2))
    test_data_output = func(test_data_input[:, 0], test_data_input[:, 1])
    test_data_output = np.expand_dims(test_data_output, axis=1)

    return train_data_input, train_data_output, test_data_input, test_data_output


def toytest():
    model = Net(2, 1)
    training_data, training_target, test_data, test_target = DataGenerator(
        100)

    train(model, training_data, training_target, test_data, test_target)


if __name__ == "__main__":
    toytest()
