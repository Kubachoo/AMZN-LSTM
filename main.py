from src.data_loader import get_stock_data
from pathlib import Path
from src.visualize import visualizeData
def main():
    # Data is in the format: ticker,start date(YYYY-MM-DD),end(YYYY-MM-DD)
    stockData = get_stock_data('AMZN','2019-05-14','2024-05-14')
    filepath = Path('/Users/kubacho/workspace/projects/LSTM/AMZN-LSTM/data/historical_data.csv')
    # Sends data as CSV file to desired path
    stockData.to_csv(filepath)

    print(stockData.head())
    print(stockData.info())
    print(stockData.describe())

    visualizeData(filepath)

if __name__ == "__main__":
    main()

