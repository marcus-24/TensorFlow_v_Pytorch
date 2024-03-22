# Use stock market values
# Use Open, High, Low, Close, values at a given day to predict the closing value of the next couple of days
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, InputLayer, LSTM
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 


DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
WIN_SIZE = 10
EPOCHS = 1000
N_FEATURES = len(DATA_COLS)


def sequential_window_dataset(series: np.ndarray, 
                              window_size: int=WIN_SIZE,
                              batch_size: int=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:, -1]))
    return ds.batch(batch_size).prefetch(1)

    
'''Load stock data'''
data = yf.download("AAPL", start="2020-01-01", end="2024-01-30")

'''Split Data'''
split_idx = int(0.75 * data.shape[0])
split_time = data.index[split_idx]

x_train = data.loc[:split_time, DATA_COLS]
train_time = x_train.index.to_numpy()
x_val = data.loc[split_time:, DATA_COLS]
val_time = x_val.index.to_numpy()

'''Normalize data'''
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

train_set = sequential_window_dataset(x_train)
val_set = sequential_window_dataset(x_val)

model = tf.keras.models.Sequential([ # input dimension (batch size, # of time steps, # of features)
  InputLayer(input_shape=(WIN_SIZE, N_FEATURES),
             batch_size=BATCH_SIZE), 
  Bidirectional(LSTM(30, return_sequences=True)),  # returns the previous time steps in a window at each neuron (100 nuerons + steps in sequence)
  Bidirectional(LSTM(30)), # 100 nuerons accepted but only the latest time output (size=1) set sent to the next layer
  Dense(1)
])

fname = os.path.join('saved_models', 'tf_best_forecast_model.h5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

optimizer = tf.keras.optimizers.legacy.SGD(lr=0.001, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model.summary()

history = model.fit(train_set, 
                    epochs=EPOCHS, 
                    validation_data=val_set,
                    callbacks=[model_checkpoint, early_stopping])


acc = history.history['mae']
val_acc = history.history['val_mae']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('# of epochs')
plt.ylabel('Loss value')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



