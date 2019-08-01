import tensorflow.keras as keras

def _block(inputs, neurons, ksize=1, act=None, pool=None):
    x = keras.layers.Conv1D(neurons, ksize)(inputs)
    x = keras.layers.BatchNormalization()(x)
    if act:
        x = keras.layers.Activation(act)(x)
    if pool is not None:
        x = keras.layers.MaxPooling1D(pool)(x)
    return x
