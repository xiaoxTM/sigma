import tensorflow as tf

from sigma import layers, colors, dbs, status
from sigma import ops

def config_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 8
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '1'
    return config

def block(x, filters, rate=None, up=False, reuse=False, name=None, scope=None):
    if name is None:
        name = ops.helper.dispatch_name('block')
    if up: # if decoder
        x = layers.convs.deconv2d(x, None, filters, 4, 2, padding='SAME',
                                  reuse=reuse, name='{}-deconv'.format(name),
                                  scope=scope)
        x = layers.norms.batch_norm(x, reuse=reuse, scope=scope,
                                 name='{}-bnorm'.format(name))
    else:
        x = layers.convs.conv2d(x, filters, 4, 2, reuse=reuse, padding='SAME',
                                name='{}-conv'.format(name), scope=scope)
        # if not the first layer of encoder
        if filters != 64:
            x = layers.norms.batch_norm(x, reuse=reuse, scope=scope,
                                       name='{}-bnorm'.format(name))
    if rate:
        x = layers.norms.dropout(x, rate, name='{}-dropout'.format(name),
                                reuse=reuse, scope=scope)
    if up: # if decoder
        x = layers.actives.relu(x, name='{}-relu'.format(name))
    else:  # elif encoder
        x = layers.actives.leaky_relu(x, name='{}-leaky-relu'.format(name))
    return x

def encoder(x, reuse=False, name=None, scope=None):
    with tf.name_scope('encoder'):
        x = block(x, 64, reuse=reuse, name=name, scope=scope)
        x = block(x, 128, reuse=reuse, name=name, scope=scope)
        x = block(x, 256, reuse=reuse, name=name, scope=scope)
        x = block(x, 512, reuse=reuse, name=name, scope=scope)
        x = block(x, 512, reuse=reuse, name=name, scope=scope)
        x = block(x, 512, reuse=reuse, name=name, scope=scope)
        x = block(x, 512, reuse=reuse, name=name, scope=scope)
        x = block(x, 512, reuse=reuse, name=name, scope=scope)
        return x

def decoder(x, nclass, rate=0.5, reuse=False, name=None, scope=scope):
    with tf.name_scope('decoder'):
        with sigma.defaults(up=True, reuse=reuse, name=name, scope=scope):
        x = block(x, 512, rate)
        x = block(x, 512, rate)
        x = block(x, 512, rate)
        x = block(x, 512)
        x = block(x, 512)
        x = block(x, 256)
        x = block(x, 128)
        x = block(x, 64)
        x = layers.convs.conv2d(x, nclass, 1, 1, padding='SAME',
                                name='output', act='tanh')
        return x


# define generator networks
def generator(ninput=99, nclass=10, scope='generator'):
    """ generator of gan model
        Attributes
        ==========
        nouts : list / tuple | int
                output shape if list / tuple
                channels of output shape if int
    """

    def _generator(shape, reuse=False):
        inputs = tf.random_normal(shape, dtype=tf.float32, name='random-noise')

        x = encoder(x, reuse=reuse, scope=scope)
        x = decoder(x, nclass, reuse=reuse, scope=scope)

        return x
    return _generator

# define discriminator networks
def discriminator(scope='discriminator'):

    def _discriminator(x, reuse=False):
        with sigma.defaults(reuse=reuse, name=name, scope=scope)
            x = block(x, 64)
            x = block(x, 128)
            x = block(x, 256)
            x = block(x, 512)
            x = block(x, 512)
            x = block(x, 512)
            x = layers.convs.conv2d(x, 1, 1, 1, act='sigmoid')
        return x

    return _discriminator


def build(ginputs, dshape, nclass=10, batch_size=256):
    discriminate = discriminator()
    generate     = generator(ginputs)
    with tf.name_scope('generator'):
        fdata = generate([batch_size, ginputs])
    with tf.name_scope('discriminator'):
        rdata = tf.placeholder(tf.float32, shape=[batch_size, *dshape], name='feed-data')
        freg = discriminate(fdata)

    with tf.name_scope('optimizer'):

        with tf.name_scope('loss'):
            # discriminator trys to recoginse real data or generated data
            # reg_rloss = layers.losses.bce(rreg, tf.ones_like(rreg))

            # blend real data with fake data (generated data)
            alpha = tf.constant(value=0.5, shape=(batch_size, 1), dtype=tf.float32)
            blabels = tf.ones_like(alpha)
            alpha = tf.reshape(alpha, [batch_size] + [1] * len(rdata.get_shape()
                                                              .as_list()[1:]))
            bdata = alpha * rdata + (1-alpha) * fdata
            breg = discriminate(bdata, reuse=True)

            reg_bloss = layers.losses.bce(breg, blabels)
            reg_floss = layers.losses.bce(freg, tf.zeros_like(freg))
            dloss = reg_floss + reg_bloss
            gloss = layers.losses.bce(freg, tf.ones_like(freg))

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
        goptimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                            beta1=0.5, beta2=0.9
                                           ).minimize(gloss, var_list=gparams,
                                                      name='ggrad')

    return goptimizer, doptimizer, gloss, dloss, rdata, fdata, bdata


def valid(session, fdata, bdata, nclass=10):
    logs = 'exp/logs/mnist/acgan/tensorboard'
    check_points = 'exp/check-points/mnist/acgan/'
    ddir = '/home/xiaox/studio/db/mnist'
    log_images = 'exp/logs/mnist/acgan/images'

    # epoch per save
    eps = 100

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

    database = dbs.mnist.sampler(ddir, True, batch_size)
    _fdata = session.run(fdata, feed_dict={flabels: crand})

    for idx, _data in enumerate(_fdata):
        plt.clf()
        plt.imshow(_data[:, :, 0], cmap='gray')
        plt.savefig(os.path.join(dirs, '{}.png'.format(idx+1)))
        print('#epoch: {}\t{}-- discriminative loss: {:+.6f} \t{}-- generative loss: {:+.6f}{}\r\b'
              .format(epoch, colors.fg.pink, _dloss,
                      colors.fg.green, _gloss, colors.reset), end='')
    print('\ndone')


def train(session, goptim, doptim, gloss, dloss,
          fdata, rdata, bdata,
          batch_size=256, nclass=10,
          epochs=10000, critic_iters=100):

    logs = 'exp/logs/mnist/acgan/tensorboard'
    check_points = 'exp/check-points/mnist/acgan/'
    ddir = '/home/xiaox/studio/db/mnist'
    log_images = 'exp/logs/mnist/acgan/images'

    # epoch per save
    eps = 100

    if logs is not None:
        tf.summary.scalar('discriminative-loss', dloss)
        tf.summary.scalar('generator-loss', gloss)
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

    database = dbs.mnist.sampler(ddir, True, batch_size)

    for epoch in range(1,epochs+1):
        # train discriminator
        status.is_training = True
        _dloss = 0
        for epoch_inside in range(critic_iters):
            samples, _ = next(database)
            _, _dloss, _summary = session.run([doptim, dloss, summarize],
                                               feed_dict={rdata: samples})
            print('\r#epoch {} / {} \t{}-- discriminative loss: {:+.6f}{}'
                  .format(epoch, epochs, colors.fg.pink, _dloss, colors.reset),
                  end='')

        for epoch_inside in range(critic_iters):
            _, _gloss = session.run([goptim, gloss])
            print('\r#epoch {} / {} \t{}-- discriminative loss: {:+.6f}{} \t{}-- generative loss: {:+.6f}{}'
                  .format(epoch, epochs, colors.fg.pink, _dloss, colors.reset,
                          colors.fg.green, _gloss, colors.reset), end='')
        print()
        # summary to tensorboard
        if logs is not None:
            writer.add_summary(_summary, global_step=epoch)

        # save to check-point and log images
        if epoch % eps == 0 and check_points is not None:
            saver.save(session, check_points, global_step=epoch)
            if log_images:
                status.is_training = False
                _fdata = session.run(fdata)
                dirs = os.path.join(log_images, 'epoch-{}'.format(int(epoch / eps)))
                os.makedirs(dirs, exist_ok=True)
                for idx, _data in enumerate(_fdata):
                    plt.clf()
                    plt.imshow(_data[:, :, 0], cmap='gray')
                    plt.savefig(os.path.join(dirs, '{}.png'.format(idx+1)))
    print('\ndone')
    if logs is not None:
        writer.close()


def run(ginputs, dshape, is_train=True, batch_size=256):
    goptim, doptim, gloss, dloss,\
        rdata, fdata, bdata = build(ginputs, dshape, 10, batch_size)
    session = tf.Session(config=config_gpu())

    session.run(tf.global_variables_initializer())

    if is_train:
        train(session, goptim, doptim, gloss, dloss,
              fdata, rdata, bdata, batch_size=batch_size, critic_iters=100)
    else:
        valid(session, fdata)
