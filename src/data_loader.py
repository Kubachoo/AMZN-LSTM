import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, start, end):
    # This downloads the historical data of the stock
    df = yf.download(ticker, start=start, end=end) 
    return df

