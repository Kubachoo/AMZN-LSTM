import yfinance as yf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Downloads the historical data of a given stock ticker within it's desired range
def get_stock_data(ticker, start, end):
    # This downloads the historical data of the stock
    df = yf.download(ticker, start=start, end=end) 
    return df

# Preprocess' and scales pricing for LSTM model
def price_scaler(info):
    df = info
    scaler = MinMaxScaler()
    importantValues = df[['Close','Open','Volume']].values
    print(scaler.fit(importantValues))
    MinMaxScaler()
    print(scaler.data_max_)
    print(scaler.transform(importantValues))


