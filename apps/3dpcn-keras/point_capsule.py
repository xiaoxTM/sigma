import tensorflow.keras as keras
import tensorflow.keras.backend as backend
from .encoder import encode
from .decoder import decode

def point_capsule_net(inputs,
                      batchsize,
                      primary_size=10,
                      num_latent=64,
                      vec_latent=64,
                      ksize=1):
    num_points = backend.int_shape(inputs)[1]

    x = encode(inputs, primary_size)
    latent_capsules = layers.capsules.dense(x, num_latent, vec_latent)

    x = decode(latent_capsules, batchsize, num_points, ksize)

    return latent_capsules, x
