from src.data_loader import get_stock_data 
from src.util import price_scaler, create_sequences 
from pathlib import Path
from src.visualize import visualizeData
from tensorflow import keras
import numpy as np

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
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(keras.layers.LSTM(64, return_sequences=False))

    # Dense layer
    model.add(keras.layers.Dense(128, activation="relu"))

    # Dropout layer
    model.add(keras.layers.Dropout(0.5))
    
    # Dense layer
    model.add(keras.layers.Dense(1))

    model.summary()
    model.compile(optimizer="adam",
                  loss="mae", 
                  metrics=[keras.metrics.RootMeanSquaredError()])

    training = model.fit(X_train, y_train, epochs=20, batch_size=32)

    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    predictions = model.predict(X_test)
    # Transform scaled predictions back to unscaled numbers
    predictions = scaler.inverse_transform(predictions)
    


if __name__ == "__main__":
    main()
 
