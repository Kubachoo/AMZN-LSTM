import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end) 
    df = df[['Close']].dropna()
    return df

