# Import

import tensorflow as tf
from keras.models import load_model
import os
import pandas as pd
from pydantic import BaseModel
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding


# Get current path location
current_path = os.getcwd()
dataset_path = os.path.join(current_path, "Dataset", "modified-dataset.csv")
df = pd.read_csv(dataset_path)

df.sample(10)

# rename df.column == "label\t" to "label"
df = df.rename(columns={"label\t": "label"})
df.columns

# change label to int type
missing_values = df["label"].isna()
df = df.drop(df[missing_values].index)

df.dtypes

print(df[df["label"] == 0].count(), df[df["label"] == 1].count(), df[df["label"] == 2].count())

# store in a dataset and random shuffle
dataset = df.values
np.random.shuffle(dataset)

# split train dev test
TRAIN_SIZE = round(len(dataset) * 0.7)
VAL_SIZE = round(len(dataset) * 0.2)
TEST_SIZE = len(dataset) - TRAIN_SIZE - VAL_SIZE

print("trainsize {}, val size {}, test size {}".format(TRAIN_SIZE, VAL_SIZE, TEST_SIZE))

# Each tweet convert to words and store in corpus
train_dataset = dataset[:TRAIN_SIZE]
val_dataset = dataset[TRAIN_SIZE: (TRAIN_SIZE+VAL_SIZE)]
test_dataset = dataset[(VAL_SIZE+ TRAIN_SIZE):]
print(len(train_dataset), len(val_dataset), len(test_dataset))
train_dataset

# train data and label
X_train = train_dataset[:, 0]
Y_train = train_dataset[:, 1]

# validation data and label
X_validation = val_dataset[:, 0]
Y_validation = val_dataset[:, 1]

# test data and label
X_test = test_dataset[:, 0]
Y_test =test_dataset[:, 1]

# Tokenizer
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=max_len)
val_sequences = tokenizer.texts_to_sequences(X_validation)
X_validation = pad_sequences(val_sequences, maxlen=max_len)

# One hot encode labels
Y_train=np.asarray(Y_train).astype(np.int)
Y_validation = np.asarray(Y_validation).astype(np.int)

temp_train = []
for val in Y_train:
  if val == 0:
    temp_train.append([1, 0, 0])
  if val == 1:
    temp_train.append([0, 1, 0])
  if val == 2:
    temp_train.append([0, 0, 1])

Y_train = temp_train

temp_val = []
for val in Y_validation:
  if val == 0:
    temp_val.append([1, 0, 0])
  if val == 1:
    temp_val.append([0, 1, 0])
  if val == 2:
    temp_val.append([0, 0, 1])

Y_validation = temp_val

Y_train=np.asarray(Y_train).astype(np.int)
Y_validation = np.asarray(Y_validation).astype(np.int)
        
class SentimentText(BaseModel):
    text_twitter: str
        
class SentimentModel:
    def __init__(self):
        self.model_fname_ =  'best_model1.hdf5'
        try:
            self.model =  load_model(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            
    def _train_model(self):
        # Building Model and Train Model
        embedding_layer = Embedding(1000, 64)
        model = Sequential()
        model.add(layers.Embedding(max_words, 20)) #The embedding layer
        model.add(layers.LSTM(15,dropout=0.5)) #Our LSTM layer
        model.add(layers.Dense(3,activation='softmax'))
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint("sentiment-analysis-CNN1D-LSTM .hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
        history = model.fit(X_train, Y_train, epochs=100,validation_data=(X_validation, Y_validation),callbacks=[checkpoint])
        return model
    
    def predict_sentiment(self, text_twitter):
        sentiment = ['Neutral','Positive','Negative']
        sequence = tokenizer.texts_to_sequences([text_twitter])
        test = pad_sequences(sequence, maxlen=max_len)
        result = sentiment[np.around(self.model.predict(test), decimals=0).argmax(axis=1)[0]]
        return result