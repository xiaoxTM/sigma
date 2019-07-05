import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import os
import sigma
from sigma import layers, ops, helpers
import os.path
import tensorflow as tf
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import exposure
import dataio

def capsule(inputs, extra_inputs, recognition_dim, generation_dim, transformation='translation', scope=None, idx=0):
    input_shape = ops.helper.norm_input_shape(inputs)
    extra_shape = ops.helper.norm_input_shape(extra_inputs)

    input_volumn = np.prod(input_shape[1:])

    gpuid = idx % 4
    with ops.core.device('/gpu:{}'.format(gpuid)):
        with sigma.defaults(scope=scope):
            x = layers.base.flatten(inputs)
            recognition = layers.convs.dense(x, recognition_dim, act='sigmoid', name='recognition-{}'.format(idx))

            probability = layers.convs.dense(recognition, 1, act='sigmoid', name='probability-{}'.format(idx))
            probability = ops.core.tile(probability, [1, input_volumn])

            if transformation == 'translation':
                learnt_transformation = layers.convs.dense(recognition, 2, name='prediction-{}'.format(idx))
                learnt_transformation = layers.math.add([learnt_transformation, extra_inputs])
            else:
                learnt_transformation = layers.convs.dense(recognition, 9, name='prediction-{}'.format(idx))
                learnt_transformation = layers.base.reshape(learnt_transformation, output_shape=[-1, 3, 3])
                learnt_transformation = layers.math.matmul([learnt_transformation, extra_inputs])
                learnt_transformation = layers.base.flatten(learnt_transformation)

            generation = layers.convs.dense(learnt_transformation, generation_dim, act='sigmoid', name='generation-{}'.format(idx))
            out = layers.convs.dense(generation, input_volumn)
            out = layers.math.mul([out, probability])
        return layers.base.reshape(out, input_shape)


def transforming_autoencoder(inputs, extra_inputs, labels, recognition_dim, generation_dim, ncapsules, transformation='translation', ngpus=4):

    with ops.core.name_scope('transformation-autoencoder'):
        capsules = []
        for i in range(ncapsules):
            cap=capsule(inputs, extra_inputs, recognition_dim, generation_dim, transformation, scope='capsule-{}'.format(i), idx=i)
            capsules.append(cap)
        inference = layers.math.add(capsules, name='autoencoder-inference')
        inference = layers.actives.sigmoid(inference)

        loss = layers.losses.mse([inference, labels])

    return inference, loss


def train(epochs=100, batchsize=100, checkpoint=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction=0.45
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    #config.gpu_options.visible_device_list='0'

    [(input_shape, extra_shape), label_shape], [train_data, valid_data, test_data]=dataio.generate_mnist_data(mode='translation')()
    train_input = np.asarray(train_data[0][0])
    train_extra = np.asarray(train_data[0][1])
    train_label = np.asarray(train_data[1])
    #valid_data = np.asarray(valid_data)
    #test_data  = np.asarray(test_data)

    inputs = layers.base.input_spec(input_shape, name='inputs')
    extras = layers.base.input_spec(extra_shape, name='extras')
    labels = layers.base.label_spec(label_shape, name='labels', dtype=ops.core.float32)

    inference, loss_op = transforming_autoencoder(inputs, extras, labels, 30, 30, 10)

    #global_step = ops.core.get_variable('global_step', initializer=0, trainable=False)
    nsamples = len(train_data[0][0])
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_step, nsamples / batchsize, 0.998)
    train_op = ops.optimizers.get('AdamOptimizer').minimize(loss_op, global_step=global_step)

    base = '/home/xiaox/studio/exp/sigma/capsules/transforming-autoencoder/mnist'

    sess = tf.Session(config=config)
    if checkpoint is not None:
        sess, saver = helpers.load(sess, checkpoint, verbose=True)

    with sess:
        valid_losses = []
        test_losses = []
        tf.global_variables_initializer().run()
        valid_feed = {inputs: valid_data[0][0],
                      extras: valid_data[0][1],
                      labels: valid_data[1]}
        test_feed  = {inputs: test_data[0][0],
                      extras: test_data[0][1],
                      labels: test_data[1]}

        indices = np.arange(0, nsamples, batchsize)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for i, idx in enumerate(indices):
                train_feed = {inputs: train_input[idx:idx+batchsize, :, :, :],
                              extras: train_extra[idx:idx+batchsize, :],
                              labels: train_label[idx:idx+batchsize, :, :, :]}

                _, loss = sess.run([train_op, loss_op], feed_dict=train_feed)
                if i % 20 == 0:
                    print("train loss for {}-th iteration: {}".format(i, loss))

            loss = sess.run(loss_op, feed_dict=valid_feed)
            valid_losses.append(loss)
            print("valid loss for {}-th epoch: {}".format(epoch, loss))
            predicts, loss = sess.run([inference, loss_op], feed_dict=test_feed)
            test_losses.append(loss)
            print("test loss  for {}-th epoch: {}".format(epoch, loss))
            if epoch % 10 == 0:
                helpers.save(sess, checkpoint, saver, True)
                for idx, (predict, label, ipts, offset) in enumerate(zip(predicts, test_data[1], test_data[0][0], test_data[0][1])):
                    predict = exposure.rescale_intensity(sk.img_as_ubyte(predict))
                    ipts = sk.img_as_ubyte(ipts)
                    label = sk.img_as_ubyte(label)
                    os.makedirs('{}/{}'.format(base, epoch), exist_ok=True)
                    images = np.concatenate((label, ipts, predict), axis=1)
                    skio.imsave('{}/{}/{}-{}-{}.png'.format(base, epoch, idx, offset[0], offset[1]), images)
        np.savetxt('valid.loss', valid_losses)
        np.savetxt('test.loss', test_losses)


if __name__ == '__main__':
    ckpt = '/home/xiaox/studio/exp/sigma/capsules/transforming-autoencoder/mnist/cache/ckpt'
    train(epochs=1001, checkpoint=ckpt)
