import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import SymbolicTensor
import tensorflow_datasets as tfds
from typing import Tuple
import numpy as np
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 1000
IMG_SHAPE = 32
N_CLASSES = 10

def scale_images(image: SymbolicTensor, label: int) -> Tuple[SymbolicTensor, int]:
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

def image_augment(image: SymbolicTensor, label: int) -> Tuple[SymbolicTensor, int]:
    image = tf.image.random_flip_left_right(image)

    return image, label

dataset, metadata = tfds.load('cifar10', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples


train_dataset = (train_dataset.map(scale_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .cache()  # cache after scaling
                              .prefetch(tf.data.experimental.AUTOTUNE)
                              .repeat()
                              .shuffle(num_train_examples)
                              .map(image_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) # no cache since transformation is random
                              .batch(BATCH_SIZE))

test_dataset = (test_dataset.map(scale_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                            .cache()
                            .batch(BATCH_SIZE))

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_SHAPE, IMG_SHAPE, 3),
                               batch_size=BATCH_SIZE),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

dot_img_file = 'model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "bestmodel.h5", save_best_only=True) # saves model after each epoch if better than best model available
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
                    validation_data=test_dataset,
                    validation_steps=math.ceil(num_test_examples/BATCH_SIZE),
                    use_multiprocessing=True,
                    workers=8,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping, model_checkpoint])


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
plt.show()