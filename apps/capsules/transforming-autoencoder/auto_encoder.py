import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, ops

import tensorflow as tf

import numpy as np

import dataio

def capsule(inputs, extra_inputs, recognition_dim, generation_dim, transformation='translation', scope=None, idx=0):
    input_shape = ops.helper.norm_input_shape(inputs)
    extra_shape = ops.helper.norm_input_shape(extra_inputs)

    input_volumn = np.prod(input_shape[1:])

    with sigma.defaults(scope=scope):
        x = layers.base.flatten(inputs)
        recognition = layers.convs.dense(x, recognition_dim, act='sigmoid', name='recognition-{}'.format(idx))

        probability = layers.convs.dense(recognition, 1, act='sigmoid', name='probability-{}'.format(idx))
        probability = ops.core.tile(probability, [1, input_volumn]) # maybe BUG here

        if transformation == 'translation':
            learnt_transformation = layers.convs.dense(recognition, 2, name='prediction-{}'.format(idx))
            learnt_transformation = layers.math.add([learnt_transformation, extra_inputs])
        else:
            learnt_transformation = layers.convs.dense(recogntion, 9, name='prediction-{}'.format(idx))
            learnt_transformation = layers.base.reshape(learnt_transformation, output_shape=[-1, 3, 3])
            learnt_transformation = layers.math.matmul([learnt_transformation, extra_inputs])
            learnt_transformation = layers.base.flatten(learnt_transformation)

        generation = layers.convs.dense(learnt_transformation, generation_dim, act='sigmoid', name='generator-{}'.format(idx))
        out = layers.convs.dense(generation, input_volumn)
        out = layers.math.mul([out, probability])
        return layers.base.reshape(out, input_shape)


def transforming_autoencoder(inputs, labels, recogntion_dim, generation_dim, ncapsules, transformation='translation'):
    inputs, extra_inputs = inputs

    with ops.core.name_scope('transforming-autoencoder'):
        capsules = [capsule(inputs, extra_inputs, recogntion_dim, generation_dim, transformation, scope='capsule-{}'.format(i), idx=i) for i in range(ncapsules)]

        inference = layers.math.add(capsules, name='autoencoder-inference')
        inference = layers.actives.sigmoid(inference)

        loss = layers.losses.mse([inference, labels])
        #metric = layers.metrics.accuracy([inference, labels])
        metric = None

    return [inference, loss, metric]


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
gpu_config.gpu_options.visible_device_list = '0,1'
gpu_config.intra_op_parallelism_threads = 1

sigma.engine.set_print(False, False)
nclass = 10
experiment, parser = sigma.build_experiment(transforming_autoencoder,
                                            dataio.generate_mnist_data('translation'),
                                            'AdamOptimizer',
                                            # filename='mnist-networks.png',
                                            batch_size=100,
                                            gpu_config=gpu_config,
                                            model_config={'recogntion_dim':30,
                                                          'generation_dim':30,
                                                          'ncapsules':10})


if __name__=='__main__':
    args = parser.parse_args()
    args.checkpoint='cache'
    # args.log='cache'
    # args.auto_timestamp = True
    experiment(args)
