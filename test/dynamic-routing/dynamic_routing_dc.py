import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, dbs, ops, engine, colors, helpers, status
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import logging
import os

import time

from tensorflow.examples.tutorials.mnist import input_data

import skimage as sk
import skimage.io as skio

from sklearn.metrics import accuracy_score, balanced_accuracy_score

tf.set_random_seed(1024)

logging.basicConfig(level=logging.INFO)

def build_func(inputs, labels, initializer='glorot_uniform', is_training=True, reuse=False):
    # inputs shape :
    #    [batch-size, 28x28]
    with sigma.defaults(reuse=reuse):
        ops.core.summarize('inputs', inputs)
        image = layers.base.reshape(inputs, [-1, 28, 28, 1], name='reshape-0')
        x = layers.convs.conv2d(image, 256, 9,
                                stride=1,
                                padding='valid',
                                act='relu',
                                name='conv2d-0',
                                weight_initializer=initializer)
        ops.core.summarize('conv2d-0', x)
        #   [batch-size, 20, 20, 256]
        #=> [batch-size, 6, 6, 256]
        x = layers.convs.conv2d(x, 32*8, 9,
                                stride=2,
                                act='relu',
                                padding='valid',
                                name='conv2d-1',
                                weight_initializer=initializer)
        ops.core.summarize('conv2d-1', x)
        #   [batch-size, 6, 6, 256]
        #=> [batch-size, 6 * 6 * 32, 8]
        #=> [batch-size, 8, 6 * 6 * 32]
        x = layers.base.reshape(x, [-1, 6*6*32, 8], name='reshape-1')
        x = layers.base.transpose(x, (0, 2, 1), name='transpose-0')
        x = layers.actives.squash(x, axis=1, epsilon=1e-9, name='squash-actives')
        ops.core.summarize('squash-actives', x)

        #  [batch_size, nrows * ncols * 32, 8]
        #=>[batch_size, 16, 10]
        # a.k.a [batch-size, 1=capsules of each channel, 16=capsule atoms/dims, 10=channels]
        # digitCapsule Layer
        random_normal = ops.initializers.get('random_normal', stddev=0.01)
        x = layers.capsules.dense(x, 10, 16, 3,
                                  share_weights=False,
                                  name='capdense-0',
                                  safe=True,
                                  epsilon=1e-9,
                                  weight_initializer=random_normal)
        ops.core.summarize('capdense-0', x)
        # norm the output to represent the existance probabilities
        # of each class
        # classification:
        #    [batch-size, caps_dims, incaps=channels]
        #=>  [batch-size, incaps=channels=10]
        classification = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-9, name='caps-norm')
        ops.core.summarize('norm-0', classification)
        class_loss = layers.losses.get('margin_loss', classification, labels, name='classification-loss')
        #loss = class_loss
        tf.summary.scalar('classification-loss', class_loss)
        ## reconstruction
        x = layers.capsules.maskout(x, index=labels, name='maskout')
        ops.core.summarize('maskout', x)
        x = layers.convs.dense(x, 512, act='relu', name='dense-0')
        ops.core.summarize('dense-0', x)
        x = layers.convs.dense(x, 1024, act='relu', name='dense-1')
        ops.core.summarize('dense-1', x)
        x = layers.convs.dense(x, 784, act='sigmoid', name='dense-2')
        ops.core.summarize('dense-2', x)
        reconstruction = layers.base.reshape(x, [-1, 28, 28, 1], name='reshape-2')
        tf.summary.image('reconstruction', reconstruction, max_outputs=10)
        recon_loss = layers.losses.mse([reconstruction, image], name='reconstruction-loss')
        tf.summary.scalar('reconstruction-loss', recon_loss)
        loss = layers.math.add([class_loss, recon_loss], [1, 0.392], name='add') #0.392 = 0.0005 * 784
        #metric = layers.metrics.accuracy([classification, labels], name='accuracy')
        ops.core.summarize('loss', loss, 'scalar')
        #ops.core.summarize('acc', metric[0], 'scalar')
    return reconstruction, classification, loss
    #return None, loss, metric

def valid(sess, dataset, inputs, labels, loss_op, classification_op, batch_size):
    steps = int(dataset.num_examples * 1.0 / batch_size)
    loss = 0
    preds = []
    truth = []
    total = 0
    for step in range(steps):
        xs, ys = dataset.next_batch(batch_size)
        size = ys.shape[0]
        feed = {inputs:xs, labels:ys, status.is_training:False}
        l, p = sess.run([loss_op, classification_op], feed_dict=feed)
        loss += (l * size)
        p = p.argmax(1)
        ys = ys.argmax(1)
        preds.extend(p)
        truth.extend(ys)
        total += size

    mean_loss = loss * 1.0 / total
    acc = accuracy_score(truth, preds)
    bac = balanced_accuracy_score(truth, preds)
    return mean_loss, acc, bac

@helpers.stampit({'checkpoint':-2}, message='capsnet_mnist')
def train(epochs=100, batchsize=100, checkpoint=None, logdir=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.visible_device_list = '0,1,2,3'
    # config.intra_op_parallelism_threads = 1
    ## allow TensorFlow put operations on CPU if operations
    ## can not been put on GPU
    config.allow_soft_placement = True

    filefolds = '/home/xiaox/studio/db/mnist'
    mnist = input_data.read_data_sets(filefolds, one_hot=True)

    inputs = layers.base.input_spec([None, 784])
    labels = layers.base.label_spec([None, 10])
    global_step = tf.Variable(0, trainable=False)

    with ops.core.device('/gpu:0'):
        reconstruction_op, classification_op, loss_op = build_func(inputs, labels)
        #metric_op, metric_update_op, metric_variable_initialize_op = metrics_op
        learning_rate = tf.train.exponential_decay(0.001, global_step, mnist.train.num_examples / batchsize, 0.998)
        train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
        #trainable_variables = tf.trainable_variables()
        #grads = tf.gradient(loss_op, trainables)
        #optimizer = ops.optimizers.get('AdamOptimizer')
        #train_op = optimizer.apply_gradients(zip(grad, trainable_variables))

    sess, saver, summarize, writer = engine.session(config=config,
                                                    checkpoint=checkpoint,
                                                    debug=False,
                                                    #address='172.31.234.152:2666',
                                                    log=logdir)

    base = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing/mnist'
    with sess:
        #tf.global_variables_initializer().run()
        #tf.local_variables_initializer().run()

        mlog = np.ones([epochs, 3])
        best_acc = 0
        steps = int(mnist.train.num_examples / batchsize)
        for epoch in range(epochs):
            start = time.time()
            trainloss = 0
            total = 0
            preds = []
            truth = []
            for step in range(steps):
                xs, ys = mnist.train.next_batch(batchsize, shuffle=True)
                size = ys.shape[0]
                train_feed = {inputs: xs, labels: ys, status.is_training:True}
                if summarize is None:
                    _, loss, pred = ops.core.run(sess, [train_op, loss_op, classification_op], feed_dict=train_feed)
                else:
                    _, loss, pred, summary = ops.core.run(sess, [train_op, loss_op, classification_op, summarize], feed_dict=train_feed)
                trainloss += (loss * size)
                total += size
                preds.extend(pred.argmax(1))
                truth.extend(ys.argmax(1))
                if summarize is not None:
                    ops.core.add_summary(writer, summary, global_step=(epoch * steps) + step)
                #if step % 100 == 0:
                #    print('train for {}-th iteration: loss:{}'.format(step, loss))
            train_loss = trainloss * 1.0 / total
            train_acc = accuracy_score(truth ,preds)
            train_bac = balanced_accuracy_score(truth, preds)
            print('train:{}, loss:{}, acc:{}, avg acc:{}'.format(epoch, train_loss, train_acc, train_bac))
            end = time.time()

            valid_loss, valid_acc, valid_bac = valid(sess, mnist.validation, inputs, labels, loss_op, classification_op, batchsize)
            print('valid:{}, loss:{}, acc:{}, avg acc:{}'.format(epoch,valid_loss, valid_acc, valid_bac))

            test_loss, test_acc, test_bac = valid(sess, mnist.test, inputs, labels, loss_op, classification_op, batchsize)
            print('test :{}, loss:{}, acc:{}, avg acc:{}'.format(epoch, test_loss, test_acc, test_bac))

            mlog[epoch, 0] = train_acc
            mlog[epoch, 1] = valid_acc
            mlog[epoch, 2] = test_acc
            if test_acc > best_acc:
                best_acc = test_acc
                helpers.save(sess, checkpoint, saver, True)
        ops.core.close_summary_writer(writer)
        np.savetxt('dc_log', mlog)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, default='checkpoint',)
parser.add_argument('--log', '-l', type=str, default=None)
parser.add_argument('--epochs', '-e', type=int, default=200)
parser.add_argument('--batch-size', '-b', type=int, default=100)

if __name__=='__main__':
    args = parser.parse_args()
    exp = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing'
    checkpoint = None
    log = None
    if args.checkpoint is not None:
        os.makedirs('{}/{}'.format(exp, args.checkpoint), exist_ok=True)
        checkpoint = os.path.join(exp, args.checkpoint, 'ckpt')
    if args.log is not None:
        os.makedirs('{}/{}'.format(exp, args.log), exist_ok=True)
        log = os.path.join(exp, args.log)
    train(args.epochs, args.batch_size, checkpoint=checkpoint, logdir=log)
