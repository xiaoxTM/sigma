import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from sigma.apps.pcn3d import dataset
from . import ops

def build_net(inputs, labels, loss='margin_loss', batch_size=32, views=4, nclass=40, reuse=False, num_gpus=4, is_training=True, **kwargs):
    if is_training:
        inputs = dataset.rotate_point_cloud(inputs, batch_size=batch_size)
        inputs = dataset.jitter_point_cloud(inputs, batch_size=batch_size)
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
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
            core.summarize('projection_0_{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_0-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_0-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 12, 512]
            #=> [batch-size, 24, 256]
            x = ops.projection_transform(x, 24, reuse=reuse, name='projection_1-{}'.format(view), act='squash')
            ops.core.summarize('projection_1_{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_1-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_1-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 24, 256]
            #=> [batch-size, 48, 128]
            #x = ops.projection_transform(x, 48, reuse=reuse, name='projection_2-{}'.format(view), act='squash')
            #ops.core.summarize('projection_2_{}'.format(view), x, reuse=reuse)
            #x = layers.norms.batch_norm(x, name='batch_norm_2-{}'.format(view), reuse=reuse)
            #core.summarize('batch_norm_2-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 48, 128]
            #=> [batch-size, 32, channels]
            print('reuse:', reuse)
            x = layers.capsules.dense(x, channels, 48, epsilon=1e-10, reuse=reuse, name='caps_fully_connected-{}'.format(view), act='squash', share_weights=False)
            core.summarize('caps_fully_connected-{}'.format(view), x, reuse=reuse)
            multiviews.append(x)
    with core.device('/gpu:{}'.format(num_gpus-1)):
        #=> [batch-size, 48, channels]
        #=> [batch-size, 48, channels*views]
        x = layers.merge.concat(multiviews, axis=2, reuse=reuse, name='concatenate')
        core.summarize('concatenate', x, reuse=reuse)
        x = layers.norms.batch_norm(x, name='batch_norm_3', reuse=reuse, is_training=is_training)
        core.summarize('batch_norm_3', x, reuse=reuse)
        #=> [batch-size, 48, channels*views]
        #=> [batch-size, 16, nclass]
        x = layers.capsules.dense(x, nclass, 16, epsilon=1e-10, reuse=reuse, name='caps_fully_connected', act='squash')
        core.summarize('caps_fully_connected', x, reuse=reuse)
        #=> [batch-size, 16, nclass]
        #=> [batch-size, nclass]
        x = layers.capsules.norm(x, axis=1, safe=True, epsilon=1e-10, name='norm', reuse=reuse)
        core.summarize('norm', x, reuse=reuse)
        loss_op = layers.losses.get(loss, x, labels)
        ops.core.summarize('train-loss', loss_op, 'scalar')
        metric_op = layers.metrics.accuracy([x, labels])
    return x, loss_op, metric_op
