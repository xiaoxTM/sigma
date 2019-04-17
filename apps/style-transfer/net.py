import sigma
from sigma import layers
from sigma.ops import initializers, core
import tensorflow as tf
import numpy as np
import h5py

feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

means = np.array([123.680, 116.779, 103.939])

def get_tv_loss(x):
    return layers.regularizers.tvr(x)

def get_content_loss(style_content, content_content):
    return tf.reduce_mean(tf.nn.l2_loss(style_content-content_content))

def get_style_loss(style_layers, content_layers):
    losses = [tf.reduce_mean(tf.nn.l2_loss(style_layers[layer]-content_layers[layer])) for layer in feature_layers]
    return tf.reduce_mean(losses)

""" calculate gram matrix given feature map
    fmap must have the shape of:
    [batch-size, rows, cols, channels]
    Return:
    [batch-size, channels, channels]
    That is, batch-size gram matrices, each of which
    has shape of [channels, channels]
"""
def gram_matrix(fmap):
    batchsize, rows, cols, nchannels = fmap.get_shape().as_list()
    x = tf.reshape(fmap, (-1, rows*cols, nchannels))
    xT= tf.transpose(x, perm=[0, 2, 1])
    x = tf.matmul(xT, x)
    #x = tf.tensordot(x, x, axes=[[2, 1], [1, 2]])
    return x / (rows * cols * nchannels)

def preprocessing(inputs):
    return inputs - means

def vgg(inputs, reuse=False):
    base = 64
    grams = {}
    x = preprocessing(inputs)
    with sigma.defaults(stride=(1,1,1,1), padding='same', act='relu',
                        kernel=(3,3), trainable=False, reuse=reuse, scope='vgg19'):
        x = layers.convs.conv2d(x, base, name='conv1_1')
        grams['conv1_1'] = gram_matrix(x)
        tf.summary.histogram('conv1', x)
        tf.summary.histogram('gram_conv1_1', grams['conv1_1'])
        x = layers.convs.conv2d(x, base, name='conv1_2')
        x = layers.pools.max_pool2d(x, pshape=(2,2), stride=2, name='pool1')

        x = layers.convs.conv2d(x, base*2, name='conv2_1')
        grams['conv2_1'] = gram_matrix(x)
        tf.summary.histogram('conv2_1', x)
        tf.summary.histogram('gram_conv2_1', grams['conv2_1'])
        x = layers.convs.conv2d(x, base*2, name='conv2_2')
        x = layers.pools.max_pool2d(x, pshape=(2,2), stride=2, name='pool2')

        x = layers.convs.conv2d(x, base*4, name='conv3_1')
        grams['conv3_1'] = gram_matrix(x)
        tf.summary.histogram('conv3_1', x)
        tf.summary.histogram('gram_conv3_1', grams['conv3_1'])
        x = layers.convs.conv2d(x, base*4, name='conv3_2')
        x = layers.convs.conv2d(x, base*4, name='conv3_3')
        x = layers.convs.conv2d(x, base*4, name='conv3_4')
        x = layers.pools.max_pool2d(x, pshape=(2,2), stride=2, name='pool3')

        x = layers.convs.conv2d(x, base*8, name='conv4_1')
        grams['conv4_1'] = gram_matrix(x)
        tf.summary.histogram('conv4_1', x)
        tf.summary.histogram('gram_conv4_1', grams['conv4_1'])
        x = layers.convs.conv2d(x, base*8, name='conv4_2')
        content = x
        tf.summary.histogram('content', content)
        x = layers.convs.conv2d(x, base*8, name='conv4_3')
        x = layers.convs.conv2d(x, base*8, name='conv4_4')
        x = layers.pools.max_pool2d(x, pshape=(2,2), stride=2, name='pool4')

        x = layers.convs.conv2d(x, base*8, name='conv5_1')
        grams['conv5_1'] = gram_matrix(x)
        tf.summary.histogram('conv5_1', x)
        tf.summary.histogram('gram_conv5_1', grams['conv5_1'])
        x = layers.convs.conv2d(x, base*8, name='conv5_2')
        x = layers.convs.conv2d(x, base*8, name='conv5_3')
        x = layers.convs.conv2d(x, base*8, name='conv5_4')

        return x, content, grams


def transblock(inputs, nouts, kernel=3, stride=1, deconv=False, act='relu', reuse=False, residual=False, output_shape=None):
    winit = initializers.truncated_normal(stddev=.1, seed=1)
    if deconv:
        x = layers.convs.deconv2d(inputs, output_shape, nouts, kernel, stride, padding='same', weight_initializer=winit, reuse=reuse)
    else:
        x = layers.convs.conv2d(inputs, nouts, kernel, stride, padding='same', weight_initializer=winit, reuse=reuse)
    x = layers.norms.instance_norm(x, reuse=reuse)
    if act is not None:
        x = layers.actives.relu(x)
    if residual:
        x = transblock(x, nouts, kernel, stride, act=None, reuse=reuse)
        x = layers.merge.add([inputs, x])
    return x


def transform(inputs, weight_path=None, reuse=False):
    base = 32
    inputs = inputs / 255.0
    x = transblock(inputs, base, 9, reuse=reuse)
    output_shape1 = core.shape(x)
    x = transblock(x, base*2, 3, 2, reuse=reuse)
    output_shape2 = core.shape(x)
    x = transblock(x, base*4, 3, 2, reuse=reuse)
    # residual network
    x = transblock(x, base*4, 3, reuse=reuse, residual=True)
    x = transblock(x, base*4, 3, reuse=reuse, residual=True)
    x = transblock(x, base*4, 3, reuse=reuse, residual=True)
    x = transblock(x, base*4, 3, reuse=reuse, residual=True)
    x = transblock(x, base*4, 3, reuse=reuse, residual=True)
    # deconvolutional network
    x = transblock(x, base*2, 3, 2, True, reuse=reuse, output_shape=output_shape2)
    x = transblock(x, base, 3, 2, True, reuse=reuse, output_shape=output_shape1)

    x = transblock(x, 3, 9, act=None, reuse=reuse)
    x = layers.merge.add([layers.actives.tanh(x), inputs])

    return layers.actives.tanh(x) * 127.5 + 255.0 / 2.0


def build(input_shape, lr, style_weight, content_weight, tv_weight):
    input_vgg = tf.placeholder(tf.float32, shape=[1]+input_shape[1:], name='input-vgg')
    input_transform = tf.placeholder(tf.float32, shape=input_shape, name='input-transform')
    _, _, style_layers = vgg(input_vgg)

    _, content_content, _ = vgg(input_transform, reuse=True)
    transformed = transform(input_transform)
    _, transform_content, transform_layers = vgg(transformed, reuse=True)

    tv_loss = get_tv_loss(transformed) * tv_weight
    content_loss = get_content_loss(content_content, transform_content) * content_weight
    style_loss = get_style_loss(style_layers, transform_layers) * style_weight

    loss = tv_loss + content_loss + style_loss

    tf.summary.scalar('total-loss', loss)
    tf.summary.scalar('style-loss', style_loss)
    tf.summary.scalar('content-loss', content_loss)
    tf.summary.scalar('total-variate-loss', tv_loss)

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_op, transformed, input_vgg, input_transform, [loss, style_loss, content_loss, tv_loss]
