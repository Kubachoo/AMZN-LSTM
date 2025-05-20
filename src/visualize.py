import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualizeData(data):
    x_data = []
    y_data = []
    df = pd.read_csv(data,index_col=0, parse_dates=True, header=[0,1])
    df.columns = [col[0] for col in df.columns]
    # Plot 1
    
    plt.figure(figsize=(14,6))
    plt.plot(df['Close'], label='Closing Price')
    plt.xlabel("Date")
    plt.ylabel("Closing price")
    plt.legend()
    plt.grid()
    plt.show()
    
