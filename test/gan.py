import tensorflow as tf
import layers

""" auxiliary classifier - gan model
"""

def generator(scope, input_size, act='leaky_relu', reuse=False):

    if input_size is None:
        input_size = 100
    inputs = tf.random_normal(input_size, dtype=tf.float32)

    input_class = tf.placeholder((1, ))
    embed = layers.convs.embedding([10, 100], input_class, reuse=reuse)
    c = layers.norm.flatten(embed)

    x = layers.merge.mul([inputs, c])

    x = layers.convs.fully_conv(scope, x, 1024, act=act, reuse=reuse)
    x = layers.convs.fully_conv(scope, x, 128 * 7 * 7, act=act, reuse=reuse)
    x = layers.base.reshape(x, [7, 7, 128])

    x = layers.pools.unpool2d(x, psize=(2, 2))
    x = layers.convs.conv2d(scope, x, (5, 5), 256, padding='same', act=act, reuse=reuse)

    x = layers.pools.unpool2d(x, psize=(2, 2))
    x = layers.convs.conv2d(scope, x, (5, 5), 128, padding='same', act=act, reuse=reuse)

    x = layers.convs.conv2d(scope, x, (2, 2), 1, padding='same', act='tanh', reuse=reuse)

    return x

def discriminator(scope, act='relu', reuse=False):

    def _discriminator(inputs):

        # block - 1
        x = layers.convs.conv2d(scope, inputs, (3, 3), 32, padding='same', weight_initializer='random_normal',
                                act='leaky_relu', name='conv-1', reuse=reuse)
        x = layers.pools.avg_pool2d(x, psize=(2, 2))
        x = layers.norm.dropout(x, 0.3)

        # block - 2
        x = layers.convs.conv2d(scope, x, (3, 3), 64, padding='same', weight_initializer='random_normal',
                                act='leaky_relu', name='conv-2', reuse=reuse)
        x = layers.pools.avg_pool2d(x, psize=(2, 2))
        x = layers.norm.dropout(x, 0.3)

        # block - 3
        x = layers.convs.conv2d(scope, x, (3, 3), 128, padding='same', weight_initializer='random_normal',
                                act='leaky_relu', name='conv-3', reuse=reuse)
        x = layers.pools.avg_pool2d(x, psize=(2, 2))
        x = layers.norm.dropout(x, 0.3)

        # block - 4
        x = layers.convs.conv2d(scope, x, (3, 3), 256, padding='same', weight_initializer='random_normal',
                                act='leaky_relu', name='conv-4', reuse=reuse)
        x = layers.pools.avg_pool2d(x, psize=(2, 2))
        x = layers.norm.dropout(x, 0.3)

        # for regression
        reg = layers.convs.fully_conv(scope, x, 1)
        # for classification
        dis = layers.convs.fully_conv(scope, x, 10, act='softmax')

        return reg, dis

    return _discriminator
