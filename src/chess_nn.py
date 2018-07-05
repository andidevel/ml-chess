"""chess_nn

Strongly based on:
http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
"""
import keras

from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D
)
from keras.models import Sequential

IMG_DIM = 227
LABELS = [
    'bb',
    'bk',
    'bn',
    'bp',
    'bq',
    'br',
    'empty',
    'wb',
    'wk',
    'wn',
    'wp',
    'wq',
    'wr'
]

NUM_CLASSES = len(LABELS)
BATCH_SIZE = 128
EPOCHS = 10

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def make_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', input_shape=(IMG_DIM, IMG_DIM, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(7, 7)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model


def prepare_samples(X, Y):
    print('X shape: ', X.shape)
    x = X.reshape(X.shape[0], IMG_DIM, IMG_DIM, 1).astype('float32')
    # Normalization
    x /= 255
    y = keras.utils.to_categorical(Y, NUM_CLASSES)
    return (x, y)


def train(model, X, y, X_test=None, y_test=None):
    x_train, y_train = prepare_samples(X, y)
    x_val = None
    y_val = None
    if X_test is not None and y_test is not None:
        x_val, y_val = prepare_samples(X_test, y_test)
    history = AccuracyHistory()
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[history]
    )
    return history


def score(model, X, y):
    x_test, y_test = prepare_samples(X, y)
    return model.evaluate(x_test, y_test, verbose=0)
