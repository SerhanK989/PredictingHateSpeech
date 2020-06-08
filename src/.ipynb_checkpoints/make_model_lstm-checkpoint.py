import numpy as np
from time import time
from numpy import newaxis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

class hate_speech_model:
    
    
    def __init__(self):
        self.model = None
        
        
    def build_model(self, num_classes = 3, ):
        model = Sequential()
        
        model.add(Embedding(41250, 100, input_length=50)) 

        model.add(LSTM(
            units=64,
            return_sequences=True))
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))

       

        model.add(LSTM(
            units=32,
            return_sequences=False))
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))
        
        

        model.add(Dense(
            units=num_classes))
        model.add(Activation("softmax"))

        adam = Adam(learning_rate = .00001)
        
        model.compile(loss="categorical_crossentropy", optimizer=adam)
        print('>> Compiled...')
        self.model = model
    
    def fit(self, X, y, batch_size=256, epochs=10, validation_split = 0.05, class_weight = [1,1,1]):
        
        earlystopping = EarlyStopping(monitor='val_loss', patience=3)
        
        self.model.fit(X,y,batch_size=batch_size, 
                       epochs=epochs, 
                       validation_split=validation_split, 
                       class_weight=class_weight,
                       callbacks = [earlystopping]
                      )
        pass
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        pass