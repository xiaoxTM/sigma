# -*- coding: utf-8 -*-

import tensorflow as tf
import sigma
from sigma import layers, colors, dbs, status, helpers, ops
import os.path
import numpy as np
import scipy.misc as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import gzip

from collections import namedtuple
Options = namedtuple('Options', ['rdata', # fed-real samples, to be fed
                                 'fdata', # generated samples
                                 'labels', # fed-real labels, to be fed
                                 'crands', # randomized labels, to be fed
                                 'noise', # random noise, to be fed
                                 'contis', # random continuous regression, to be fed
                                 'dloss', # discriminative loss
                                 'gloss']) # generative loss

from utils import tsne

import logging
#logging.basicConfig(level=logging.DEBUG)

def save(objs, filename):
    f = gzip.open(filename, 'wb')
    pickle.dump(objs, f)
    f.close()


def config_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 8
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '5'
    return config


# define discriminator networks
def discriminator(nconts=2, nclass=10, scope='discriminator'):
    act = 'leaky_relu'
    winit = 'he_uniform'
    padding = 'same'

    def _discriminator(x, reuse=False):
        with sigma.defaults(padding=padding, weight_initializer=winit, act=act,
                            reuse=reuse, scope=scope):
            x = layers.convs.conv2d(x, 64, 5, 2, act=None, name='conv2d-1')
            # batch-size x 14 x 14 x 64 => batch-size x 7 x 7 x 128
            x = layers.convs.conv2d(x, 128, 5, 2, name='conv2d-2')
            # batch-size x 7 x 7 x 128 => batch-size x 6272
            x = layers.base.flatten(x, name='flatten')
            # 6272 => 1024
            x = layers.convs.dense(x, 1024, name='dense')

            # batch-size x 1024 => batch-size x 1 for regression
            reg = layers.convs.dense(x, 1, name='regression')

            x = layers.convs.dense(x, 128, name='classification')
            cat = layers.convs.dense(x, nclass, name='categorical')
            con = layers.convs.dense(x, nconts, act='sigmoid', name='continuous')
        return reg, cat, con

    return _discriminator


# define generator networks
def generator(ninput=99, nclass=10, scope='generator'):
    """ generator of gan model
        Attributes
        ==========
        nouts : list / tuple | int
                output shape if list / tuple
                channels of output shape if int
    """
    winit = 'he_uniform'
    act = 'relu'
    padding = 'same'

    def _generator(x, reuse=False):
        with sigma.defaults(weight_initializer=winit, reuse=reuse,
                            act=act, scope=scope, padding=padding):
            x = layers.convs.dense(x, 1024, act=None, name='dense-1')
            x = layers.norms.batch_norm(x, name='batchnorm-1')
            # batch-size x 1024 => batch-size x (7*7*128)
            x = layers.convs.dense(x, 7*7*128, act=None, name='dense-2')
            x = layers.norms.batch_norm(x, name='batchnorm-2')
            # batch-size x (7 * 7 * 128) => batch-szie x 7 x 7 x 128
            x = layers.base.reshape(x, (-1, 7, 7, 128), name='reshape')

            # deconv2d(inputs, output_shape, output_channels, kernel_size, stride)
            #    if output_shape = None, output shape will be determined by
            #       input shape and strides
            # batch-size x 7 x 7 x 128 => batch-size x 14 x 14 x 64
            x = layers.convs.deconv2d(x, None, 64, 5, 2, name ='conv2d-1')
            x = layers.norms.batch_norm(x, name='batchnorm-3')
            # batch-size x 14 x 14 x 128 => batch-size x 28 x 28 x 1
            x = layers.convs.deconv2d(x, None, 1, 3, 2, act='sigmoid', name='conv2d-2')
        return x
    return _generator


def build(ginputs, dshape, nconts, nclass=10, batch_size=256):
    discriminate = discriminator(nconts, nclass)
    generate     = generator(ginputs, nclass)
    # latent noise
    noise = tf.placeholder(tf.float32, shape=[batch_size, ginputs], name='noise')
    # latent labels
    labels = tf.placeholder(tf.int32, shape=[batch_size, nclass], name='labels')
    # latent continuous
    contis = tf.placeholder(tf.float32, shape=[batch_size, nconts], name='continuous')
    # random labels for generated samples
    crands = tf.placeholder(tf.int32, shape=[batch_size, nclass],name='random-class')
    fdata = generate(tf.concat([noise, tf.cast(crands, dtype=tf.float32) / nclass,
                                contis], axis=-1))

    rdata = tf.placeholder(tf.float32, shape=[batch_size, *dshape], name='data')
    freg, fcat, fcon = discriminate(fdata)
    rreg, rcat,   _  = discriminate(rdata, reuse=True)

    reg_rloss = layers.losses.bce([rreg, tf.ones_like(rreg)], axis=None)
    reg_floss = layers.losses.bce([freg, tf.zeros_like(freg)], axis=None)

    cat_rloss = layers.losses.cce([rcat, labels], axis=None)
    cat_floss = layers.losses.cce([fcat, crands], axis=None)

    con_floss = layers.losses.mse([fcon, contis], axis=None)
    # con_floss ignored
    # con_floss = layers.losses.mse([fcon, contis])
    reg_gloss = layers.losses.bce([freg, tf.ones_like(freg)], axis=None)

    reg_dloss = (reg_rloss + reg_floss) / 2
    cat_dloss = (cat_rloss + cat_floss) / 2

    dloss = con_floss + reg_dloss + cat_dloss
    gloss = con_floss + reg_gloss + cat_dloss

    #for collection in tf.get_default_graph().get_all_collection_keys():
    #    print(collection)

    gparams = tf.get_collection('generator')
    if len(gparams) == 0:
        print('ERROR: generator has no parameters to optimize. \
              make sure you set `scope` to layers with trainable parameters')
    print('parameters for generator training')
    for gp in gparams:
        print('    ', gp.name, gp.get_shape().as_list())
    dparams = tf.get_collection('discriminator')
    if len(dparams) == 0:
        print('ERROR: discriminator has no parameters to optimize.\
              make sure you set `scope` to layers with trainable parameters')
    print('parameters for discriminator training')
    for dp in dparams:
        print('    ', dp.name, dp.get_shape().as_list())

    doptimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                        beta1=0.5, beta2=0.9
                                       ).minimize(dloss, var_list=dparams,
                                                  name='dgrad')
    goptimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.5, beta2=0.9
                                       ).minimize(gloss, var_list=gparams,
                                                  name='ggrad')
    options = Options(rdata, fdata, labels, crands, noise, contis, dloss, gloss)
    return goptimizer, doptimizer, options


def generate_sample(session, options, samples,
                    labels, crands, noise, contis, dirs):
    status.is_training = False
    fdata = session.run(options.fdata,
                        feed_dict={options.rdata: samples,
                                   options.labels: labels,
                                   options.crands: crands,
                                   options.noise: noise,
                                   options.contis: contis})
    os.makedirs(dirs, exist_ok=True)
    for idx, data in enumerate(fdata):
        sm.imsave(os.path.join(dirs, '{}.png'.format(idx+1)), data[:, :, 0])


def train(session, goptim, doptim, options,
          batch_size=256, nclass=10,
          epochs=100000, critic_iters=3):

    logs = 'exp/logs/mnist/acgan/tensorboard'
    check_points = 'exp/check-points/mnist/acgan/'
    ddir = '/home/xiaox/studio/db/mnist'
    log_images = 'exp/logs/mnist/acgan/images'

    # epoch per save
    eps = 10

    if logs is not None:
        tf.summary.scalar('discriminative-loss', options.dloss)
        tf.summary.scalar('generator-loss', options.gloss)
        summarize = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs, session.graph)

    # if set to check point, configure checkpoint options
    if check_points is not None:
        saver = tf.train.Saver()
        # try to store from last newest experiment state, if exists
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(check_points))
        if ckpt and ckpt.model_checkpoint_path:
            print('{}load check point from {}{}{}'
                 .format(colors.fg.cyan, colors.fg.red,
                         ckpt.model_checkpoint_path, colors.reset))
            saver.restore(session, ckpt.model_checkpoint_path)

    database = dbs.images.mnist.sampler(ddir, True, batch_size, onehot=True, nclass=nclass)
    rsamples, rlabels = next(database)
    rnoise = np.random.normal(0, 1, size=options.noise.get_shape().as_list())
    rcontis = np.random.normal(0, 1, size=options.contis.get_shape().as_list())
    rcrands = np.random.uniform(0, nclass, size=(batch_size)).astype(np.int32)
    rcrands = helpers.one_hot(rcrands, nclass)

    for epoch in range(1,epochs+1):
        status.is_training = True
        dloss = None
        for epoch_inside in range(critic_iters):
            samples, labels = next(database)
            noise = np.random.normal(0, 1, size=rnoise.shape)
            crands = np.random.uniform(0, nclass, size=[batch_size])
            crands = helpers.one_hot(crands.astype(np.int32), nclass)
            contis = np.random.normal(0, 1, size=rcontis.shape)
            _, dloss, summary = session.run([doptim, options.dloss, summarize],
                                            feed_dict={options.rdata: samples,
                                                       options.labels: labels,
                                                       options.crands: crands,
                                                       options.noise: noise,
                                                       options.contis: contis
                                                       })
            print('\r#epoch {} / {} \t{}-- discriminative loss: {:+.6f}{}'
                  .format(epoch, epochs, colors.fg.pink, dloss, colors.reset),
                  end='')

        # summary to tensorboard
        if logs is not None:
            writer.add_summary(summary, global_step=epoch)

        for epoch_inside in range(critic_iters):
            noise = np.random.normal(0, 1, size=rnoise.shape)
            crands = np.random.uniform(0, nclass, size=[batch_size])
            crands = helpers.one_hot(crands.astype(np.int32), nclass)
            contis = np.random.normal(0, 1, size=rcontis.shape)
            #gloss = session.run(options.gloss,
            _, gloss = session.run([goptim, options.gloss],
                                   feed_dict={options.rdata: samples,
                                              options.labels: labels,
                                              options.crands: crands,
                                              options.noise: noise,
                                              options.contis: contis})
            print('\r#epoch {} / {} \t{}-- discriminative loss: {:+.6f}{}'
                  '\t{}-- generative loss: {:+.6f}{}'
                  .format(epoch, epochs, colors.fg.pink, dloss, colors.reset,
                          colors.fg.green, gloss, colors.reset), end='')


        # save to check-point and log images
        if epoch % eps == 0 and check_points is not None:
            saver.save(session, check_points, global_step=epoch)
            if log_images:
                dirs = os.path.join(log_images, 'epoch-{}'.format(int(epoch / eps)))
                os.makedirs(dirs, exist_ok=True)
                generate_sample(session, options, rsamples,
                                rlabels, rcrands, rnoise, rcontis, dirs)
        if critic_iters > 1:
            print()
    print('\ndone')
    if logs is not None:
        writer.close()


def run(ginputs, dshape, is_training=True, batch_size=256):
    #sigma.engine.set_print(True, False)
    goptim, doptim, options = build(ginputs, dshape, 2, 10, batch_size)
    #layers.core.export_graph("graph.png")
    session = tf.Session(config=config_gpu())

    session.run(tf.global_variables_initializer())

    train(session, goptim, doptim, options, batch_size=batch_size,
          epochs=10000, critic_iters=3)
