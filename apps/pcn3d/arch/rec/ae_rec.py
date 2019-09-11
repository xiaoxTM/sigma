import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from sigma.apps.pcn3d import dataset
from . import ops

def block(x, channels, is_training, idx, act='relu', reuse=False):
    x = layers.convs.dense(x, channels, name='fully_connected-{}'.format(idx), reuse=reuse)
    ops.core.summarize('fully_connected-{}'.format(idx), x, reuse=reuse)

    x = layers.norms.batch_norm(x, name='batch_norm_recon-{}'.format(idx), reuse=reuse, is_training=is_training)
    ops.core.summarize('batch_norm_recon-{}'.format(idx), x, reuse=reuse)

    x = eval('layers.actives.{}(x, reuse=reuse, name="active_{}")'.format(act, idx))
    ops.core.summarize('active_{}'.format(idx), x, reuse=reuse)

    return x

def build_net(inputs, labels, loss='categorical_cross_entropy', batch_size=32, nclass=40, num_gpus=4, views=4, reuse=False, is_training=True, **kwargs):
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
    if is_training:
        inputs = dataset.rotate_point_cloud(inputs, batch_size=batch_size)
        inputs = dataset.jitter_point_cloud(inputs, batch_size=batch_size)
    begin = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose')
    #        [batch-size, 3, 2048]
    core.summarize('inputs', begin, reuse=reuse)
    gpus = range(num_gpus)
    multiviews = []
    channels = nclass#int(2048 / views)
    for view in range(views):
        with core.device('/gpu:{}'.format(view%num_gpus)):
            #=> [batch-size, 3, 2048]
            #=> [batch-size, 12, 512]
            x = ops.projection_transform(begin, 12, reuse=reuse, name='projection_0-{}'.format(view), act='squash')
            core.summarize('projection_0-{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_0-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_0-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 12, 512]
            #=> [batch-size, 24, 256]
            x = ops.projection_transform(x, 24, reuse=reuse, name='projection_1-{}'.format(view), act='squash')
            ops.core.summarize('projection_1-{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_1-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_1-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 24, 256]
            #=> [batch-size, 48, 128]
            #x = ops.projection_transform(x, 48, reuse=reuse, name='projection_2-{}'.format(view), act='squash')
            #ops.core.summarize('projection_2_{}'.format(view), x, reuse=reuse)
            #x = layers.norms.batch_norm(x, name='batch_norm_2-{}'.format(view), reuse=reuse, is_training=is_training)
            #core.summarize('batch_norm_2-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 48, 128]
            #=> [batch-size, 32, channels]
            x = ops.permutation_transform(x, channels, 48, name='permutation_transform-{}'.format(view), act='squash', reuse=reuse)
            core.summarize('permutation_transform-{}'.format(view), x, reuse=reuse)
            multiviews.append(x)
    with core.device('/gpu:{}'.format(num_gpus-1)):
        #=> [batch-size, 48, channels]
        #=> [batch-size, 48, channels*views]
        x = layers.merge.concat(multiviews, axis=2, reuse=reuse, name='concatenate')
        core.summarize('concatenate', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_3', reuse=reuse, is_training=is_training)
        #core.summarize('batch_norm_3', x, reuse=reuse)
        #=> [batch-size, 48, channels*views]
        #=> [batch-size, 16, nclass]
        x = layers.capsules.dense(x, nclass, 16, epsilon=1e-10, reuse=reuse, name='caps_fully_connected', act='squash')
        core.summarize('caps_fully_connected', x, reuse=reuse)
        # for reconstruction
        # for recognition
        #=> [batch-size, 16, nclass]
        #=> [batch-size, nclass] (recog)
        recog = layers.capsules.norm(x, axis=1, safe=True, epsilon=1e-10, name='norm', reuse=reuse)
        core.summarize('norm', recog, reuse=reuse)

        recog_loss_op = layers.losses.get(loss, recog, labels, reuse=reuse, name='recognition-loss')
        ops.core.summarize('recognition-loss', recog_loss_op, 'scalar', reuse=reuse)
        metric_op = layers.metrics.accuracy([recog, labels], reuse=reuse, name='accuracy')

        #=> [batch-size, 16, nclass]
        #=> [batch-size, 16 * nclass]
        if is_training:
            recon = layers.base.maskout(x, labels, name='maskout', reuse=reuse)
            ops.core.summarize('maskout', recon, reuse=reuse)
        else:
            recon = layers.base.maskout(x, name='maskout_valid')
        #x = layers.convs.dense(recon, 256, name='fully_connected-1', reuse=reuse)
        #ops.core.summarize('fully_connected-1', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_recon_1', reuse=reuse, is_training=is_training)
        #ops.core.summarize('batch_norm_recon_1', x, reuse=reuse)
        #x = layers.convs.dense(x, 512, name='fully_connected-2', reuse=reuse, act='relu')
        #ops.core.summarize('fully_connected-2', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_recon_2', reuse=reuse, is_training=is_training)
        #ops.core.summarize('batch_norm_recon_2', x, reuse=reuse)
        #x = layers.convs.dense(x, 1024, name='fully_connected-3', reuse=reuse, act='relu')
        #ops.core.summarize('fully_connected-3', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_recon_3', reuse=reuse, is_training=is_training)
        #ops.core.summarize('batch_norm_recon_3', x, reuse=reuse)
        #x = layers.convs.dense(x, 2048, name='fully_connected-4', reuse=reuse, act='relu')
        #ops.core.summarize('fully_connected-4', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_recon_4', reuse=reuse, is_training=is_training)
        #ops.core.summarize('batch_norm_recon_4', x, reuse=reuse)
        x = block(recon, 512, is_training, 0, reuse=reuse)
        x = block(x, 1024, is_training, 1, reuse=reuse)
        x = block(x, 2048, is_training, 2, reuse=reuse)
        x = block(x, 2048*3, is_training, 3, reuse=reuse, act='tanh')

        #x = layers.convs.dense(x, 2048 * 3, name='fully_connected-5', reuse=reuse, act=None)
        #ops.core.summarize('fully_connected-5', x, reuse=reuse)
        #x = layers.actives.tanh(x, name='actives-final', reuse=reuse)
        #ops.core.summarize('actives-final', x, reuse=reuse)

        #x = layers.norms.batch_norm(x, name='batch_norm_recon_5', reuse=reuse, is_training=is_training)
        #ops.core.summarize('batch_norm_recon_5', x, reuse=reuse)

        x = layers.base.reshape(x, [-1, 3, 2048], name='reshape', reuse=reuse) 
        recon_loss_op = layers.losses.mse([x, begin], reuse=reuse, name='reconstruction-loss')
        ops.core.summarize('reconstruction-loss', recon_loss_op, 'scalar', reuse=reuse)
        loss_op = layers.math.add([recog_loss_op, recon_loss_op], [1, 0.0005 * nclass * 3], name='add', reuse=reuse)
        ops.core.summarize('total-loss', loss_op, 'scalar', reuse=reuse)
    return recog, loss_op, metric_op
