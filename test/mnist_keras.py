import tensorflow as tf
from tensorflow.contrib.keras import layers, models, losses, callbacks
from tensorflow.contrib.keras import preprocessing
ImageDataGenerator = preprocessing.image.ImageDataGenerator
from tensorflow.contrib.keras import datasets
mnist = datasets.mnist
from tensorflow.contrib.keras import utils
to_categorical = utils.to_categorical
from tensorflow.contrib.keras import backend as K
# from tensorflow.examples.tutorials.mnist import input_data
import logging
# logging.basicConfig(level=logging.DEBUG)
# import numpy as np
# np.random.seed(1000)
import os
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
# print(sys.path)
# from sigma import layers
import numpy as np

class HistoryRecorder(callbacks.Callback):
    def __init__(self, recordnames, filename):
        """
            Record history
            Attributes
            ----------
            recordnames : string or list of string
                          metrics or losses to record
            eps : int, can be None
                  epoch for save. will not save if None
            epm : int, can be None
                  after epm epochs to send message
            mode : string ['accumulate', 'segment']
                   mode to send record
                   'accumulate' -- from epoch 0 to current epoch
                   'segment' -- send epm epochs (that is, from current-epm epochs to current epochs)
        """
        assert isinstance(recordnames, (list, tuple, str)), 'recorder names can accept list or tuple type only!'
        self.filename = filename
        if isinstance(recordnames, str):
            recordnames = [recordnames]
        self.recordnames = recordnames
        super(HistoryRecorder, self).__init__()

    def on_train_begin(self, logs={}):
        self.records = []

    def on_epoch_end(self, epoch, logs={}):
        self.records.append([logs.get(name) for name in self.recordnames])

    def on_train_end(self, logs={}):
        np.savetxt(self.filename, self.records, header='|'.join(self.recordnames))

# import time

def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, translate, scale,
            shuffle=True):
    if set_name == 'train':
        (X, Y), _ = get_mnist_dataset()
    elif set_name == 'test':
        _, (X, Y) = get_mnist_dataset()

    image_gen = ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate
    )
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen

def model_keras():
    inputs = layers.Input((28, 28, 1), name='input')
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv11')(inputs)
    x = layers.Activation('relu', name='conv11_relu')(x)
    x = layers.BatchNormalization(name='conv11_bn')(x)

    # conv12
    x = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12')(x)
    x = layers.Activation('relu', name='conv12_relu')(x)
    x = layers.BatchNormalization(name='conv12_bn')(x)

    # conv21
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv21')(x)
    x = layers.Activation('relu', name='conv21_relu')(x)
    x = layers.BatchNormalization(name='conv21_bn')(x)

    # conv22
    x = layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22')(x)
    x = layers.Activation('relu', name='conv22_relu')(x)
    x = layers.BatchNormalization(name='conv22_bn')(x)

    # out
    x = layers.GlobalAvgPool2D(name='avg_pool')(x)
    x = layers.Dense(10, name='fc1')(x)
    x = layers.Activation('softmax', name='out')(x)

    return models.Model(inputs=inputs, outputs=x)

def accuracy(logits, labels):
    y = tf.argmax(tf.nn.softmax(logits), axis=-1)
    _y = tf.argmax(labels, axis=-1)
    acc = tf.cast(tf.reduce_sum(tf.where(tf.equal(y, _y), tf.ones_like(y), tf.zeros_like(y))), dtype=tf.float32)
    return tf.div(acc, tf.cast(tf.shape(y)[0], dtype=tf.float32))

if __name__ == '__main__':

    mode = 'bilinear'

    model = 'keras'

    batch_size = 50
    n_train = 60000
    n_test = 10000
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_test / batch_size))

    logloss = HistoryRecorder(['val_loss', 'val_acc'], 'test/data/mnist/keras/bilinear/scaled_glorot_uniform')

    # loss = layers.losses.cce(out, y)
    # # acc = accuracy(out, y)
    #
    # losses = []
    loss = losses.categorical_crossentropy
    train_gen = get_gen(
        'train', batch_size=batch_size,
        scale=(0.25, 2.5), translate=0.2,
        shuffle=True
    )

    test_gen = get_gen(
        'test', batch_size=batch_size,
        scale=(0.25, 2.5), translate=0.2,
        shuffle=False
    )

    model = model_keras()
    model.compile('adam', loss, metrics=['acc'])

    model.fit_generator(
        train_gen, steps_per_epoch=steps_per_epoch,
        epochs=10, verbose=1,
        validation_data=test_gen, validation_steps=validation_steps,
        callbacks=[logloss]
    )
