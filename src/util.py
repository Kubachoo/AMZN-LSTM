import yfinance as yf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# Preprocess' and scales pricing for LSTM model
def price_scaler(info):
    df = info
    scaler = MinMaxScaler()
    important_values = df[['Close','Open','Volume']].values
    transformed_values = scaler.fit_transform(important_values)
    scaled_df = pd.DataFrame(transformed_values, columns=['Close', 'Open', 'Volume'], index = df.index)
    return scaled_df, scaler


def create_sequences(df, window_size):
    df_as_np = df.to_numpy()
    # input data
    X = []
    # Output value
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        input_value = df_as_np[i+window_size]
        y.append(input_value)

    print(np.array(X))
    return np.array(X), np.array(y)
