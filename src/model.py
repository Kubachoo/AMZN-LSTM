from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from main.py import X_train,y_train
import matplotlib.pyplot as plt

def train_model():
    model = Sequential([
        LSTM(60,activation='relu', input_shape=(window_size,1)),
        Dense(1)
    ]) 
    
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train, y_train, epochs=20,batch_size=32,verbose=1)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    plt.figure(figsize=(10,6))

