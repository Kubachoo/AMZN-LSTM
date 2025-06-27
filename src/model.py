from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model():
   model = Sequential([
        LSTM(60,activation='relu', input_shape=(window_size,1)),
        Dense(1)
    ]) 

    model.compile(optimizer='adam',loss='mse')
    model.fit(
