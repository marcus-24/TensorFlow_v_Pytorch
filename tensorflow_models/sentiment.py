import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import os
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt

EPOCHS = 1000
BATCH_SIZE = 64
VOCAB_SIZE = 1000
MAX_LEN = 150
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
EMBEDDING_DIM = 16
OOV_TOKEN = '<OOV>'
N_CLASSES = 3



'''Load data'''
fname = os.path.join('data', 'Tweets.csv')
df = pd.read_csv(fname)

'''Create feature and label arrays'''
tweets = df['text'].to_list()
labels = df['airline_sentiment'].to_list()
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
labels_int = np.array([label_map[label] for label in labels])  # convert to numpy for network compatibility

'''Split data'''
training_size = int(0.7 * len(tweets))

training_tweets = tweets[:training_size]
testing_tweets = tweets[training_size:]
training_labels = labels_int[:training_size]
testing_labels = labels_int[training_size:]

'''Transform data'''
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_tweets)
training_sequences = tokenizer.texts_to_sequences(training_tweets)
training_padded = pad_sequences(training_sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_tweets)
testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_DIM)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('log_files', f'tf_senitment_{time_now}')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

print(model.summary())

fname = os.path.join('saved_models', 'tf_best_sentiment_model.h5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True)  # saves model after each epoch if better than best model available
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(training_padded,
                    training_labels,
                    steps_per_epoch=math.ceil(len(training_padded)/BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=(testing_padded, testing_labels),
                    validation_steps=math.ceil(len(testing_padded)/BATCH_SIZE),
                    use_multiprocessing=True,
                    workers=8,
                    callbacks=[early_stopping, model_checkpoint, tensorboard_callback])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

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



