import sigma
from sigma import ops, layers

from .block import _block

def decode(name, inputs, batchsize, num_points=2048, grid_size=2, channels=1, reuse=False, is_training=True):
    with sigma.defaults(reuse=reuse):
        # inputs should have shape of
        #=> [batch-size, num-latent, vec-latent]
        outs = []
        num_latent, vec_latent = ops.core.shape(inputs)[1:]
        parts = int(num_points / num_latent)
        for idx in range(parts):
            # random to generate grid with shape of
            #=>  [batch-size, num-latent, grid-size]
            grid = layers.base.random_spec([batchsize,
                                            num_latent,
                                            grid_size],
                                           minval=0.0,
                                           maxval=1.0,
                                           name='{}_grid'.format(name))

            # inputs: [batch-size, num-latent, vec-latent]
            #   grid: [batch-size, num-latent, grid-size]
            #=>    x: [batch-size, num-latent, vec-latent+grid-size]
            x = layers.merge.concat([inputs, grid], axis=2, name='{}_concat_{}'.format(name, idx))
            #ops.core.summarize('{}_concat_{}'.format(name, idx), x)

            #      x: [batch-size, num_latent, vec-latent+grid-size]
            #=>    x: [batch-size, num_latent, vec-latent]
            x = _block('{}_1_{}'.format(name, idx), x, vec_latent, channels=channels, act='relu', is_training=is_training)

            #      x: [batch-size, num-latent, vec_latent]
            #=>    x: [batch-size, num-latent, vec_latent/2]
            x = _block('{}_2_{}'.format(name, idx), x, int(vec_latent / 2), channels, act='relu', is_training=is_training)

            #      x: [batch-size, num-latent, vec_latent/2]
            #=>    x: [batch-size, num-latent, vec_latent/4]
            x = _block('{}_3_{}'.format(name, idx), x, int(vec_latent / 4), channels, act='relu', is_training=is_training)

            #      x: [batch-size, num-latent, vec_latent/4]
            #=>    x: [batch-size, num-latent, 3]
            x = layers.convs.conv1d(x, 3, channels, name='{}_conv1d_{}'.format(name, idx))
            #ops.core.summarize('{}_conv1d_{}'.format(name, idx), x)

            x = layers.actives.tanh(x, name='{}_tanh_{}'.format(name, idx))
            #ops.core.summarize('{}_tanh_{}'.format(name, idx), x)

            outs.append(x)

        # outs: [batch-size, num-latent, 3] * parts
        #=>  x: [batch-size, num-latent * parts, 3]
        # num_points = num_latent * parts
        x = layers.merge.concat(outs, axis=1, name='{}_concat'.format(name))
        #ops.core.summarize('{}_concat'.format(name), x)
        return x
