#! /usr/bin/env python3

import sys
import os
import time
import os.path

import numpy as np

import tensorflow as tf

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../../'))
sys.path.append(root)

from auto_encoder import point_capsule_rec
from sigma import layers, engine, ops, status, helpers
from dataset import shapenet_part
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--num-latent', default=64, type=int)
parser.add_argument('--vec-latent', default=64, type=int)
parser.add_argument('--primary-size', default=16, type=int)
parser.add_argument('--nclass', default=16, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--logdir', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--dataset', default='/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)

def cutpoints(points, rows, num_points):
    indices = tf.range(rows, dtype=tf.int32)
    indices = tf.random.shuffle(indices)
    points = tf.gather(points, indices)
    slices = np.arange(num_points, dtype=np.int32)
    points = tf.gather(points, slices)
    def _cut():
        return points
    return _cut

def parse(num_points):
    def _parse(record):
        features = tf.parse_single_example(
                record,
                features={
                    'rows': tf.FixedLenFeature([], tf.int64),
                    'cols': tf.FixedLenFeature([], tf.int64),
                    'points': tf.FixedLenFeature([], tf.string),
                    'part_labels': tf.FixedLenFeature([], tf.string),
                    'category_label': tf.FixedLenFeature([], tf.int64),
                    }
                )
        points = tf.cast(tf.decode_raw(features['points'], tf.float64), tf.float32)
        category = tf.cast(features['category_label'], tf.int32)
        rows = tf.cast(features['rows'], tf.int32)
        cols = tf.cast(features['cols'], tf.int32)
        points = tf.reshape(points, [rows, cols])
        points = tf.cond(tf.greater(rows, num_points), cutpoints(points, rows, num_points), lambda :points)
        points = tf.reshape(points, [num_points, 3])
        return points, tf.one_hot(category, 16)
    return _parse

def load_record(filename, batchsize=2, shuffle=True, num_points=2048):
    filename_queue = tf.train.string_input_producer([filename], shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return parse(num_points)(serialized_example)

def train_net(args):
    engine.set_print(None)
    train_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized.tfrecord'
    valid_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_valid.tfrecord'
    tests_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_test.tfrecord'

    filename = tf.placeholder(tf.string, shape=[])
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse(args.num_points))
    dataset = dataset.shuffle(1000).batch(args.batch_size)
    dataset = dataset.repeat(args.epochs)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    #ops.core.summarize('inputs', inputs)
    def _build_net(inputs, reuse, is_training):
        return point_capsule_rec(inputs,
                                 is_training,
                                 args.primary_size,
                                 args.vec_latent,
                                 args.nclass,
                                 args.channels,
                                 reuse=reuse)

    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)
    with ops.core.device('/gpu:0'):
        trainp = _build_net(inputs, reuse=False, is_training=True)
        train_loss_op = layers.losses.get('margin_loss', trainp, labels)
        train_metric = layers.metrics.accuracy([trainp, labels])
        train_metric_op, train_metric_update_op, train_metric_initialize_op = train_metric
        train_iters = 10000
        learning_rate = tf.train.exponential_decay(0.001, global_step, train_iters, 0.9)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        with ops.core.control_dependencies(update_ops):
            train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(train_loss_op, global_step=global_step)

        validp = _build_net(inputs, reuse=True, is_training=False)
        valid_loss_op = layers.losses.get('margin_loss', validp, labels)
        valid_metric = layers.metrics.accuracy([validp, labels])
        valid_metric_op, valid_metric_update_op, valid_metric_initialize_op = valid_metric

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    sess, saver, summarize, writer = engine.session(checkpoint=args.checkpoint,
                                                    config=config,
                                                    debug=args.debug,
                                                    address=args.address,
                                                    log=args.logdir)

    base = '/home/xiaox/studio/exp/3dpcn'
    #layers.core.export_graph('auto_encoder.png')
    with sess:
        losses =  np.zeros((args.epochs, 4))
        for epoch in range(args.epochs):
            start = time.time()
            ops.core.run(sess, [train_metric_initialize_op, iterator.initializer], feed_dict={filename: train_filename})
            for iters in range(train_iters):
                _, loss, _, summary = ops.core.run(sess, [train_op, train_loss_op, train_metric_update_op, summarize])
                accuracy = ops.core.run(sess, train_metric_op)
                ops.core.add_summary(writer, summary, global_step=(epoch*train_iters)+iters)
                if iters % 10 == 0:
                    print('train for {}-th iteration: loss: {}, accuracy: {}'.format(iters, loss, accuracy))
            end = time.time()
            print('time cost:', end-start)
            # validation
            valid_loss = []
            valid_acc = []
            ops.core.run(sess, [valid_metric_initialize_op, iterator.initializer], feed_dict={filename: valid_filename})
            for iters in range(valid_iters):
                loss, _ = ops.core.run(sess, [valid_loss_op, valid_metric_update_op])
                accuracy = ops.core.run(sess, valid_metric_op)
                valid_loss.append(loss)
                valid_acc.append(accuracy)
            vloss = np.mean(valid_loss)
            vacc = np.mean(valid_acc)
            losses[epoch][0] = vloss
            losses[epoch][1] = vacc
            print('valid for {}-th epoch: loss:{}, accuracy: {}'.format(epoch, vloss, vacc))
            # test
            test_loss = []
            test_acc = []
            ops.core.run(sess, [valid_metric_initialize_op, iterator.initializer], feed_dict={filename: tests_filename})
            for iters in range(test_iters):
                loss, _ = ops.core.run(sess, [valid_loss_op, valid_metric_update_op])
                accuracy = ops.core.run(sess, valid_metric_op)
                test_loss.append(loss)
                test_acc.append(accuracy)
                #if epoch % 20 == 0:
                #    for idx, (ipt, tst) in enumerate(zip(testx, recons)):
                #        os.makedirs('{}/{}/{}'.format(base, epoch, iters), exist_ok=True)
                #        np.savetxt('{}/{}/{}/{}-input.txt'.format(base, epoch, iters, idx), ipt)
                #        np.savetxt('{}/{}/{}/{}-recst.txt'.format(base, epoch, iters, idx), tst)
            tloss = np.mean(test_loss)
            tacc = np.mean(test_acc)
            losses[epoch][2] = tloss
            losses[epoch][3] = tacc
            print('test for {}-th epoch: loss:{}, accuracy: {}'.format(epoch, tloss, tacc))
            if epoch % 10 == 0:
                helpers.save(sess, args.checkpoint, saver, True, global_step=epoch)
        np.savetxt('losses.log', losses)
        ops.core.close_summary_writer(writer)

def eval_net(args):
    inputs, x = build_net(args)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.phase == 'train':
        train_net(args)
    elif args.phase == 'eval':
        eval_net(args)
    else:
        raise ValueError('mode must be either `train` or `eval`. given `{}`'.format(args.phase))
    #with tf.Session() as sess:
    #    #coord = tf.train.Coordinator()
    #    #coord = None
    #    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #    #inputs, labels = load_record('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_valid.tfrecord',
    #    #                             batchsize=2)
    #    filenames = tf.placeholder(tf.string, shape=[])
    #    valid = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_valid.tfrecord'
    #    train = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation.tfrecord'
    #    dataset = tf.data.TFRecordDataset(filenames)
    #    dataset = dataset.map(parse(2048))
    #    dataset = dataset.shuffle(100).batch(2)
    #    dataset = dataset.repeat(2)
    #    iterator = dataset.make_initializable_iterator()
    #    #iterator = dataset.make_one_shot_iterator()
    #    next_data = iterator.get_next()
    #    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    #    sess.run(iterator.initializer, feed_dict={filenames:train})
    #    #dataset = tf.data.TFRecordDataset(train_file)
    #    #dataset.map(parse(2048))
    #    #dataset = dataset.shuffle(100).batch(2)
    #    #iterator = dataset.make_initializable_iterator()
    #    #inputs, labels = iterator.get_next()
    #    print('done')
    #    while True:
    #        try:
    #            #inputs_batch, labels_batch = sess.run([inputs, labels])
    #            inputs_batch, labels_batch = sess.run(next_data)
    #            print(inputs_batch, labels_batch)
    #        except tf.errors.OutOfRangeError:
    #            break
    #    #coord.request_stop()
    #    #coord.join(threads)
