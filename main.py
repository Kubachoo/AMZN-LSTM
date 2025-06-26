from src.data_loader import get_stock_data 
from src.util import price_scaler, create_sequences
from pathlib import Path
from src.visualize import visualizeData
def main():
    # Data is in the format: ticker,start date(YYYY-MM-DD),end(YYYY-MM-DD)
    stock_data = get_stock_data('AMZN','2019-05-14','2024-05-14')
    filepath = Path('/Users/kubacho/workspace/projects/LSTM/AMZN-LSTM/data/historical_data.csv')

    # Sends data as CSV file to desired path
    stock_data.to_csv(filepath)
    # Prints basic information on CSV file
    print(stock_data.head())
    print(stock_data.info())
    print(stock_data.describe())
    
    # Visualizes CSV file in order to spot outliers
    visualizeData(filepath)
    # This scales the data within the range of (0,1) 
    price_scaler(stock_data)
    
    create_sequences(stock_data,60)

if __name__ == "__main__":
    main()
 
