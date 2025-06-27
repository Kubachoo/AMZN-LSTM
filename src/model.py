from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from main.py import X_train,y_train
def train_model():
   model = Sequential([
        LSTM(60,activation='relu', input_shape=(window_size,1)),
        Dense(1)
    ]) 

    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train, y_train, epochs=100,batch_size=32,verbose=1)
