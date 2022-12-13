#import data form projet.py
from projet import split_data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import MinMaxScaler



print("Loading models...")
#load models
LSTMmodel = torch.load("LSTMmodel.pt")
GRUmodel = torch.load("GRUmodel.pt")
print("Models loaded")


#make a function for evaluation
def model_evaluation(pred,gold, model):
    #calculate mse between gold and pred
    mse = np.mean((pred - gold)**2)
    return mse

mse_lstm=list()
mse_gru=list()


#create a function to randomly select multiple stocks from input/Data/Stocks to test
def random_stock_test(plot=False):
    #get list of all stocks
    stock_list = os.listdir("input/Data/Stocks")
    #select random stock
    stock = random.choice(stock_list)
    #load data
    df = pd.read_csv("input/Data/Stocks/" + stock)
    #get data from 2010 to 2016
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    price = df[['Open']]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Open'] = scaler.fit_transform(price['Open'].values.reshape(-1,1))

    lookback = 20 # choose sequence length
    x_train_, y_train_, x_test_, y_test_ = split_data(price, lookback)

    #print('x_train.shape = ',x_train.shape)
    #print('y_train.shape = ',y_train.shape)

    #convert to tensor
    x_train = torch.from_numpy(x_train_).type(torch.Tensor)
    y_train = torch.from_numpy(y_train_).type(torch.Tensor)
    x_test = torch.from_numpy(x_test_).type(torch.Tensor)
    y_test = torch.from_numpy(y_test_).type(torch.Tensor)


    #predict
    pred_lstm = LSTMmodel(x_test)
    pred_gru = GRUmodel(x_test)

    #convert to numpy
    pred_lstm = pred_lstm.detach().numpy()
    pred_gru = pred_gru.detach().numpy()

    # #reshape
    # pred_lstm = pred_lstm.reshape(-1,1)
    # pred_gru = pred_gru.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)

    #inverse transform
    pred_lstm = scaler.inverse_transform(pred_lstm)
    pred_gru = scaler.inverse_transform(pred_gru)
    y_test = scaler.inverse_transform(y_test)

    #calculate mse
    mse_lstm = model_evaluation(pred_lstm,y_test,"LSTM")
    mse_gru = model_evaluation(pred_gru,y_test,"GRU")

    #plot
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(y_test, label='True')
        plt.plot(pred_lstm, label='LSTM Prediction')
        plt.plot(pred_gru, label='GRU Prediction')
        plt.title(stock)
        plt.legend()
        plt.show()
    
    return mse_lstm, mse_gru


#test 5 stocks
for i in range(5):
    mse_lstm, mse_gru = random_stock_test(True)
    print("LSTM mse: ", mse_lstm)
    print("GRU mse: ", mse_gru)

