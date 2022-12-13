import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



#importing data
data = pd.read_csv('./input/Data/Stocks/goog.us.txt')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
price = data[['Open']]
#print(data.index.min(), data.index.max())
#data.head()

# date_split = '2016-01-01'
# train = price[:date_split]
# test = price[date_split:]

# print(train.shape, test.shape)

# #data visualization
# plt.figure(figsize=(15, 5))
# plt.plot(train, label='train')
# plt.plot(test, label='test')
# plt.legend(loc='best')
# plt.title('Opening Prices')
# plt.grid(True)
# #plt.show()


#data preprocessing with pytorch
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Open'] = scaler.fit_transform(price['Open'].values.reshape(-1,1))

#function to split data into train and test
def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(price, lookback)

#print('x_train.shape = ',x_train.shape)
#print('y_train.shape = ',y_train.shape)


x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)




#LSTM model
import torch.nn as nn
import torch.nn.functional as F

class StockModelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(StockModelLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):

        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        #print("x shape:",x.shape)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        

        #print("h0 shape:",h0.shape)
        #print("c0 shape:",c0.shape)

        #out, _ = self.lstm(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
            
        return out

#GRU model
class StockModelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(StockModelGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out

            
#training
import torch.optim as optim

LSTMmodel = StockModelLSTM(1, 32, 3, 1)
GRUmodel = StockModelGRU(1, 32, 3, 1)


train_losses_lstm = []
test_losses_lstm = []
train_losses_gru = []
test_losses_gru = []
tot_time_lstm = 0
tot_time_gru = 0

def evaluate_model(model,num_epochs,model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #calculate time it takes to train each model
    global tot_time_lstm, tot_time_gru

    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        
        #training
        #now=time.time()
        model.train()
        for t in range(num_epochs):
            output = model(x_train)
            loss = criterion(output, y_train_lstm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # if model_name == "LSTM":
            #     tot_time_lstm += time.time()-now
            # else:
            #     tot_time_gru += time.time()-now
        

        #testing
        model.eval()
        for t in range(num_epochs):
            output = model(x_test)
            loss = criterion(output, y_test_lstm)
            test_loss += loss.item()
        
        #train_loss = train_loss / len(x_train)
        #test_loss = test_loss / len(x_test)
        if model_name == "LSTM":
            train_losses_lstm.append(train_loss)
            test_losses_lstm.append(test_loss)
        else:
            train_losses_gru.append(train_loss)
            test_losses_gru.append(test_loss)

        #print all information pertaining to the model
        print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            test_loss
            ))
    
    
# #plotting losses
# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Testing loss')
# plt.legend(frameon=False)
# plt.show()

# num_epochs = 100
# model = StockModel(1, 32, 2,1)
# criterion = torch.nn.MSELoss(reduction='mean')
# optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# import time
# hist = np.zeros(num_epochs)
# start_time = time.time()
# lstm = []
# for t in range(num_epochs):
#     y_train_pred = model(x_train)    
#     loss = criterion(y_train_pred, y_train_lstm)
#     print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#     optimiser.zero_grad()
#     loss.backward()
#     optimiser.step()
    
# training_time = time.time()-start_time
# print("Training time: {}".format(training_time))

# #plotting losses
# plt.plot(hist, label="Training loss")
# plt.legend()





# #plotting losses
# figure, axis = plt.subplots(1, 2, figsize=(10, 5))
# axis[0].plot(train_losses_lstm, label='Training loss LSTM')
# axis[0].plot(test_losses_lstm, label='Testing loss LSTM')

# axis[1].plot(train_losses_gru, label='Training loss GRU')
# axis[1].plot(test_losses_gru, label='Testing loss GRU')

# axis[0].legend(frameon=False)
# axis[1].legend(frameon=False)
# plt.show()




# #testing
# y_test_pred_lstm = LSTMmodel(x_test)
# y_test_pred_gru = GRUmodel(x_test)
# y_test_pred_lstm = y_test_pred_lstm.detach().numpy()
# y_test_pred_gru = y_test_pred_gru.detach().numpy()


# #plot each model on seperate graphs
# figure, axis = plt.subplots(1, 2, figsize=(10, 5))
# axis[0].plot(y_test_lstm, label='Actual')
# axis[0].plot(y_test_pred_lstm, label='Predicted by LSTM')

# axis[1].plot(y_test_lstm, label='Actual')
# axis[1].plot(y_test_pred_gru, label='Predicted by GRU')

# axis[0].legend(frameon=False)
# axis[1].legend(frameon=False)

# plt.show()


#make a function that graphs all relevant information
def graph_all():
    global y_test
    #testing
    y_test_pred_lstm = LSTMmodel(x_test)
    y_test_pred_gru = GRUmodel(x_test)
    y_test_pred_lstm = y_test_pred_lstm.detach().numpy()
    y_test_pred_gru = y_test_pred_gru.detach().numpy()

    #plotting losses
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].plot(train_losses_lstm, label='Training loss LSTM')
    axis[0].plot(test_losses_lstm, label='Testing loss LSTM')

    axis[1].plot(train_losses_gru, label='Training loss GRU')
    axis[1].plot(test_losses_gru, label='Testing loss GRU')

    axis[0].legend(frameon=False)
    axis[1].legend(frameon=False)
    plt.show()

    #inverse transform
    y_test_pred_lstm = scaler.inverse_transform(y_test_pred_lstm)
    y_test_pred_gru = scaler.inverse_transform(y_test_pred_gru)
    y_test = scaler.inverse_transform(y_test)



    #plot each model on seperate graphs
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].plot(y_test, label='Actual')
    axis[0].plot(y_test_pred_lstm, label='Predicted by LSTM')
    
    axis[1].plot(y_test, label='Actual')
    axis[1].plot(y_test_pred_gru, label='Predicted by GRU')

    axis[0].legend(frameon=False)
    axis[1].legend(frameon=False)

    plt.show()


def main():
    #evaluate both models
    
    now = time.time()
    evaluate_model(LSTMmodel,30,"LSTM")
    tot_time_lstm = time.time() - now
    now = time.time()
    evaluate_model(GRUmodel,30,"GRU")
    tot_time_gru = time.time() - now
    #print time taken to train
    print("Time taken to train and test LSTM: ", tot_time_lstm)
    print("Time taken to train and test GRU: ", tot_time_gru)


    #graph all relevant information
    graph_all()



    

#save model so it can be used later
torch.save(LSTMmodel, "LSTMmodel.pt")
torch.save(GRUmodel, "GRUmodel.pt")

# #load model
# LSTMmodel = torch.load("LSTMmodel.pt")
# GRUmodel = torch.load("GRUmodel.pt")


if __name__=="__main__":
    main()




