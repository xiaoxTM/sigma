from __future__ import print_function
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import os
import os.path
import argparse
from argparse import ArgumentParser
import net
import tensorflow as tf
from scipy import misc as sm
from scipy import io as sio
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import sigma
from sigma import helpers, layers, dbs
#import inspect
import h5py

exp = '/home/xiaox/studio/exp/sigma/capsules/style-transfer'

class PositiveCheckAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(PositiveCheckAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0:
            raise ValueError('weight must be greater than zero, given {}'.format(values))
        setattr(namespace, self.dest, values)


class PathCheckAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(PathCheckAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None and not os.path.exists(values):
            raise ValueError('given {} not exists'.format(values))
        setattr(namespace, self.dest, values)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoints', type=str, action=PathCheckAction, required=True)
    parser.add_argument('--style', type=str, action=PathCheckAction, required=True)
    parser.add_argument('--content', type=str, action=PathCheckAction, required=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--weight-path', type=str, action=PathCheckAction, default=None)
    parser.add_argument('--epochs', type=int, action=PositiveCheckAction, default=2)
    parser.add_argument('--batch-size', type=int, action=PositiveCheckAction, default=1)
    parser.add_argument('--content-weight', type=float, action=PositiveCheckAction, default=7.5e0)
    parser.add_argument('--style-weight', type=float, action=PositiveCheckAction, default=1e2)
    parser.add_argument('--tv-weight', type=float, action=PositiveCheckAction, default=2e2)
    parser.add_argument('--learning-rate', type=float, action=PositiveCheckAction, default=1e-3)

    return parser


def load_weights_from_mat(session):
    weights = {}
    graph = session.graph
    params = graph.get_collection('vgg19')
    for param in params:
        name = param.name
        _, layername, layertype, variable = name.split('/')
        weights['{}_{}'.format(layername, variable)] = param
    data = sio.loadmat('cache/weights/imagenet-vgg-verydeep-19.mat')
    mean = data['normalization'][0][0][0]
    mean = np.mean(mean, axis=(0, 1))
    matweights = data['layers'][0]
    for i in range(matweights.shape[0]):
        if len(matweights[i][0][0]) <= 2:
            continue
        ltype = matweights[i][0][0][2][0]
        if isinstance(ltype, str) and ltype == 'conv':
            w, b = matweights[i][0][0][0][0]
            w = np.transpose(w, (1, 0, 2, 3))
            b = b.reshape(-1)
            lname = matweights[i][0][0][3][0]
            wname = '{}_weight:0'.format(lname)
            if wname not in weights.keys():
                continue
            print('initializing {} from MatLab file'.format(wname))
            aw = tf.assign(weights[wname], w)
            bname = '{}_bias:0'.format(lname)
            print('initializing {} from MatLab file'.format(bname))
            ab = tf.assign(weights[bname], b)
            session.run([aw, ab])


def load_weights(session):
    weights = {}
    graph = session.graph
    params = graph.get_collection('vgg19')
    for param in params:
        name = param.name
        _, optype, *_ = name.rsplit('_')
        if optype not in ['pool']:
            weights[name] = param
    with h5py.File('cache/weights/vgg19_weights_init.h5', 'r') as f:
        for k, v in f.items():
            if 'pool' not in k:
                weight_names = v.attrs['weight_names']
                for weight_name in weight_names:
                    block, wb, _ = weight_name.decode('utf8').rsplit('_', 2)
                    if 'W' in wb:
                        name = 'vgg19/{}/conv2d/weight:0'.format(block)
                    else:
                        name = 'vgg19/{}/conv2d/bias:0'.format(block)
                    assign = tf.assign(weights[name], np.asarray(v[weight_name]))
                    session.run(assign)


def load(style, content, shape):
    style_image, _ = dbs.images.load_image(style, size=shape[1:], mode='bilinear')
    if os.path.isdir(content):
        content_images = dbs.images.make_generator_from_list(content, batch_size=shape[0], size=shape[1:], mode='bilinear')
    elif os.path.isfile(content):
        content_images, _ = dbs.images.load_image(content, size=shape[1:], mode='bilinear')
    return style_image, content_images


def train(args):
    input_shape = [args.batch_size, 256, 256, 3]
    #sigma.set_print(None)
    train_op, transformed, input_vgg, input_transform, losses = net.build(input_shape,
                                                                          args.learning_rate,
                                                                          args.style_weight,
                                                                          args.content_weight,
                                                                          args.tv_weight)
    #helpers.export_graph('model.png')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '3'
    config.intra_op_parallelism_threads = 8
    style_image, (generator, _, iterations) = load(args.style, args.content, input_shape)
    generator = generator('input', 'label')
    samples, _ = dbs.images.load_from_dir('dtrain/samples', size=input_shape[1:], mode='bilinear')
    name = args.style.split('/')[-1][:-4]
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('training with style:', name)
        load_weights_from_mat(sess)
        os.makedirs(os.path.join(args.checkpoints, name, 'model.ckpt'), exist_ok=True)
        sess, saver = helpers.load(sess, os.path.join(args.checkpoints, name))
        # helpers.export_weights('cache/weights/style-transfer.sigma', sess)
        # helpers.import_weights('cache/weights/style-transfer.sigma', sess)
        #summarize = tf.summary.merge_all()
        #writer = tf.summary.FileWriter('cache/logs', sess.graph)
        for epoch in range(1, args.epochs+1):
            while True:
                content_images, iteration = next(generator)
                _, _losses = sess.run([train_op, losses],
                                       feed_dict={input_vgg:style_image,
                                                  input_transform:content_images['input']
                                                 })
                #writer.add_summary(_summarize, global_step=(epoch*iterations+iteration))
                print('\rIter:{:4d}/{:4d}\t total:{:.4f}\t--style:{:.4f}\t--content:{:.4f}\t--tv:{:.4f}'
                      .format(iteration, iterations, _losses[0], _losses[1], _losses[2], _losses[3]), end='')
                if iteration == iterations:
                    break
                    # test on the first-10 images
            print('')
            _transformed = sess.run(transformed, feed_dict={input_transform:samples})
            sess, saver = helpers.save(sess, os.path.join(args.checkpoints, name, 'model.ckpt'), saver, global_step=epoch)
            os.makedirs('cache/epochs/{}/{}'.format(name, epoch), exist_ok=True)
            for idx, images in enumerate(zip(samples, _transformed.astype(np.uint8))):
                    sm.imsave('cache/epochs/{}/{}/{}.png'.format(name, epoch, idx), helpers.stack(images, interval=10, value=[0,0,0]))
        print('\n>>> DONE')
        #writer.close()


def evaluate(args):
    input_shape = [args.batch_size, 256, 256, 3]
    #sigma.set_print(None)
    train_op, transformed, input_vgg, input_transform, losses = net.build(input_shape,
                                                                          args.learning_rate,
                                                                          args.style_weight,
                                                                          args.content_weight,
                                                                          args.tv_weight)
    #helpers.export_graph('model.png')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '1'
    config.intra_op_parallelism_threads = 8

    style_image, generator = load(args.style, args.content, input_shape)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess, saver = helpers.load(sess, args.checkpoints)
        if isinstance(generator, (list, tuple)):
            generator, _, iterations = generator
            while True:
                (content_images, _), iteration = next(generator)
                _transformed = sess.run(input_transform,
                                        feed_dict={input_vgg:style_image,
                                                   input_transform:content_images
                                       })
                os.makedirs('{}/cache/iter/{}'.format(exp, epoch), exist_ok=True)
                for idx, images in enumerate(zip(content_images, _transformed.astype(np.uint8))):
                    sm.imsave('{}/cache/iter/{}/{}.png'
                              .format(exp, iteration, idx),
                              helpers.stack(images, interval=10, value=[0,0,0]))
                if iteration == iterations:
                    break
        else:
            _transformed = sess.run(input_transform,
                                    feed_dict={input_vgg:style_image,
                                               input_transform:generator})
            sm.imsave(os.path.join(exp, 'cache/result.png'),
                      helpers.stack([generator, _transformed.astype(np.uint8)],
                                     interval=10, value=[0, 0, 0]))
        print('\n>>> DONE')
        writer.close()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        evaluate(args)
