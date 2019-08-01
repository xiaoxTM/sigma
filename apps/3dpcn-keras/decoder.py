import tensorflow.keras as keras
import tensorflow.keras.backend as backend

def decode(inputs, batchsize, num_points=2048, grid_size=2, ksize=1):
    outs = []
    num_latent, vec_latent = backend.int_shape(inputs)[1:]
    parts = int(num_points / num_latent)

    for idx in range(parts):
        grid = backend((batchsize, num_latent, grid_size))
        x = keras.layers.Concatenate(axis=2)([inputs, grid])

        x = _block(x, vec_latent, ksize=ksize, act='relu')
        x = _block(x, int(vec_latent / 2), ksize, act='relu')
        x = _block(x, int(vec_latent / 4), ksize, act='relu')
        x = keras.layers.Conv1D(3, ksize)(x)
        x = keras.layers.Activation('tanh')
        outs.append(x)

    return keras.layers.Concatenate(axis=1)(outs)
