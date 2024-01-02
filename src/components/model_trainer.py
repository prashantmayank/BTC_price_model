import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional, SpatialDropout1D, MaxPooling1D
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from src.exception import CustomException
from src.logger import logging



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,df):
        try:
            df=pd.read_csv(df)
            X=df.cleaned_tweets
            y = pd.get_dummies(df['sentiment']).values
            num_classes = df['sentiment'].nunique()

            seed = 38 # fix random seed for reproducibility
            np.random.seed(seed)
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.2,
                                                                stratify=y,
                                                                random_state=seed)
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            max_features = 20000
            tokenizer = Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(list(X_train))
            X_train = tokenizer.texts_to_sequences(X_train)
            X_test = tokenizer.texts_to_sequences(X_test)
            max_words = 30
            X_train = sequence.pad_sequences(X_train, maxlen=max_words)
            X_test = sequence.pad_sequences(X_test, maxlen=max_words)
            batch_size = 128
            epochs = 10

            max_features = 20000
            embed_dim = 100

            np.random.seed(seed)
            K.clear_session()
            model = Sequential()
            model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))    
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            

            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                          epochs=epochs, batch_size=batch_size, verbose=2)
            
            # predict class with test set
            y_pred_test =  np.argmax(model.predict(X_test), axis=1)

            return (y_test,y_pred_test)
        
        except Exception as e:
            raise CustomException(e,sys)
        