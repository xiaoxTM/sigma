import tensorflow.keras as keras
from tensorflow.keras import backend
from .block import _block
from sigma import ops

def encode(inputs, primary_size):
    x = _block(inputs, 64, act='relu')
    x = _block(x, 128, act='relu')

    N = backend.int_shape(x)[1]
    capsules = [_block(x, 1024, act=None, pool=N) for _ in range(primary_size)]

    x = keras.layers.Concatenate(axis=-1)(capsules)
    x = keras.layers.Permute((0, 2, 1))(x)

    return ops.squash(safe=True)(x)
