import numpy as np
from time import time
from numpy import newaxis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

class hate_speech_model:
#this is a pretty simple LSTM that's used to detect hate speech
    
    
    def __init__(self):
        """
        Input: None
        Output: None
        
        This just sets up a dummy variable to save the model in.
        """
        self.model = None
        
        
    def build_model(self, num_classes = 3, embedding_size = 500):
        """
        Input:
            num_classes: the number of classes to predict. We have 3 classes, so the default is 3
            embedding_size: the size of the embedding to train. 
            learning_rate: the learning rate for our optimizer
        Output:
            None
            
        This initializes our model in a class attribute to train later
        It starts with an embedding layer, then moves onto two LSTM Layers followed by a dense layer
        We add dropout after each LSTM Layer to prevent overfitting
        We also use adam as our optimizer with a default learning rate of .00001
        The loss function we minimize is categorical cross entropy
        """
        model = Sequential()
        
        model.add(Embedding(41250, embedding_size, input_length=50, learning_rate = .00001))
        
        model.add(LSTM(
            units=128,
            return_sequences=True,
        ))
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))

        model.add(LSTM(
            units=64,
            return_sequences=False,
        ))
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))
     
        
        model.add(Dense(
            units=num_classes,
        ))
        model.add(Activation("softmax"))

        adam = Adam(learning_rate = .00001)
        
        model.compile(loss="categorical_crossentropy", optimizer=adam)
        print('>> Compiled...')
        self.model = model

    
    def fit(self, X, y, batch_size=256, epochs=10, validation_split = 0.05, class_weight = None):
        """
        Input:
            X: the Input data to train our model with
            y: the predictions we fit to
            batch_size: the batch size for the model
            epochs: the number of epochs to run training for
            validation_split: what percentage of our data is used to monitor our models training progress
            class_weight: the class weights to use while training
        Output:
            None
        
        This just fits our model with the parameters we provide
        We stop when our validation loss doesn't improve for 5 epochs to prevent overfitting
        """
        
        earlystopping = EarlyStopping(monitor='val_loss', patience=5)
        
        self.model.fit(X,y,
                       batch_size=batch_size, 
                       epochs=epochs, 
                       validation_split=validation_split, 
                       class_weight=class_weight,
                       callbacks = [earlystopping]
                      )

    
    
    def predict(self, X):
        """
        Input:
            X: the Input data to run through our model
        Ouput:
            The result of running X through our model
        """
        return self.model.predict(X)