from src.data_loader import get_stock_data 
from src.util import price_scaler, create_sequences 
from pathlib import Path
from src.visualize import visualizeData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

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
    scaled_data,scaler = price_scaler(stock_data)
    # This function creates the time sequences used for forecasting    
    X_train, y_train = create_sequences(scaled_data,50)
    X_test, y_test = create_sequences(scaled_data,50)


    # Model training   
    model = Sequential([
        LSTM(60,activation='relu', input_shape=(50,1)),
        Dense(1)
    ]) 
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train, y_train, epochs=20,batch_size=32,verbose=1)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    plt.figure(figsize=(10,6))
    

if __name__ == "__main__":
    main()
 
