from .. import helpers, colors, dbs
from ..ops import helper
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import math


def intsize(x, cminus=False):
    if x > 0:
        return int(np.log10(x)) + 1
    elif x == 0:
        return 1
    else:
        if cminus:
            return int(np.log10(-x)) + 2
        else:
            return int(np.log10(-x)) + 1


def predict(input_shape,
            predop=None,
            axis=None,
            dtype='int64',
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'flatten', reuse)
    if predop is None:
        predop = tf.argmax
    elif isinstance(predop, str):
        if predop == 'argmax':
            predop = tf.argmax
        elif predop == 'argmin':
            predop = tf.argmin
        else:
            raise ValueError('`predop` must be one of `argmax` or `argmin`.'
                             ' given {}'.format(predop))
    elif not callable(predop):
        raise TypeError('`predop` must be type of None or str or callable.'
                        ' given {}'.format(type(predop)))
    if axis is None:
        axis = 0
    elif axis < 0:
        axis = helper.normalize_axes(input_shape, axis)
    rank = len(input_shape)
    axis = rank - 1
    # if rank == 1: #[depth]
    #     axis = 0
    # elif rank == 2: #[batch-size, depth]
    #     axis = 1
    # elif rank == 3: #[batch-size, units, depth]
    #     axis = 2
    # elif rank == 4: #[batch-size, rows, cols, depth]
    #     axis = 3
    def _predict(x):
        with ops_scope:
            return predop(x, axis, name)
    return _predict


def session(target='',
            graph=None,
            config=None,
            initializers=None,
            checkpoints=None,
            logs=None,
            verbose=True):
    sess = tf.Session(target, graph, config)
    sess.run(tf.global_variables_initializer())
    if initializers is not None:
        sess.run(initializers)
    ans = {'session': sess}
    if checkpoints is not None:
        sess, saver = helpers.load(sess, checkpoints, verbose=verbose)
        ans['session'] = sess
        ans['saver'] = saver
    if logs is not None:
        summarize = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs, sess.graph)
        ans['summarize'] = summarize
        ans['writer'] = writer
    return ans


def line(iterable,
         brief=False,
         nprompts=10,
         epochs=None,
         multiplier=1,
         enum=False,
         timeit=False,
         show_total_time=False,
         accuracy=5,
         message=None,
         nc='x',
         cc='+'):
    """ show line message
    """
    if epochs is None:
        try:
            epochs = len(iterable)
        except TypeError:
            raise TypeError('getting the length of `iterable` failed')
    if message is None:
        message = '@'
    nc = nc.encode('utf8')
    cc = cc.encode('utf8')
    step = (nprompts / (epochs * multiplier))
    _prompt = np.asarray([nc] * nprompts, dtype=np.string_)
    epochsize = intsize(epochs)
    beg = None
    elapsed = None
    if brief:
        if timeit:
            spec = '\r{} [{{:{}}}, {{:3}}%] -- {{:.{}}}(s) {{}}' \
                   .format(message, nprompts, accuracy)
        else:
            spec = '\r{} [{{:{}}}, {{:3}}%] {{}} '.format(message, nprompts)
    else:
        if timeit:
            spec = '\r{} [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' -- {{:.{}}}(s) {{}}'.format(message,
                                                 nprompts,
                                                 epochsize,
                                                 epochsize,
                                                 accuracy)
        else:
            spec = '\r{} [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%] {{}} ' \
                   .format(message, epochsize, epochsize, nprompts)
    def _line():
        totaltime = 0
        prev = 0
        for idx, epoch in enumerate(iterable):
            time_beg = timer()
            if enum:
                epoch = [idx, epoch]
            ret = (yield epoch)

            block_beg = idx * step
            totaltime += (timer() - time_beg)
            idx += 1
            elapsed = totaltime / idx
            if ret is None:
                ret = ''
            if block_beg > nprompts:
                block_beg = nprompts - step
            block_end = int(min(max(block_beg + step, 1), nprompts))
            block_beg = int(block_beg)
            if _prompt[block_beg] == nc or prev != block_beg:
                _prompt[block_beg:block_end] = cc
                if prev != block_beg:
                    _prompt[prev:block_beg] = cc
            else:
                _prompt[block_beg:block_end] = nc
            prev = block_beg
            percentage = int(idx * 100 / epochs)
            prompt = _prompt[:block_end+1].tostring().decode('utf-8')
            # print('specification:', spec)
            if brief:
                if timeit:
                    message = spec.format(prompt, percentage, elapsed, ret)
                else:
                    message = spec.format(prompt, percentage, ret)
            else:
                if timeit:
                    message = spec.format(prompt, idx, epochs, percentage, elapsed, ret)
                else:
                    message = spec.format(prompt, idx, epochs, percentage, ret)
            print(message, end='')
        if show_total_time and timeit:
            print('\nTotal time elapsed:{}(s)'.format(totaltime))
        else:
            print()
    return _line


def _loss_filter(save='all'):
    def _max(x, y):
        if y is None:
            return True
        else:
            return x > y

    def _min(x, y):
        if y is None:
            return True
        else:
            return x < y

    def _all(x, y):
        return True

    if save == 'all':
        return _all
    elif save == 'max':
        return _max
    elif save == 'min':
        return _min
    else:
        raise ValueError('`save` must be `all`, `max` or `min`. given {}'
                         .format(save))


def run(x, xtensor, optimizer, loss,
        y=None,
        ytensor=None,
        nclass=None,
        metrics=None,
        epochs=1000,
        batch_size=32,
        shuffle=True,
        valids=None,
        graph=None,
        config=None,
        checkpoints=None,
        logs=None,
        save='all'):
    """ run to train networks
        Attributes
        ==========
            x : np.ndarray or list / tuple
                samples to train
            xtensor : tf.placeholder
                      input placeholder of the network
            optimizer : tf.optimizer
                        optimizer to update network parameter
            loss : tf.Tensor
                   objectives network tries to minimize / maximize
            y : np.ndarray or list / tuple / None
                ground truth for x.
            ytensor : tf.placeholder / None
                      y placeholder of the network
            nclass : int / None
                     number of class for classification
                     Note, None indicates no one_hot op when generate
                     (sample, label) pairs
            metrics : tf.Op / None
                      measurement of result accuracy
            epochs : int
                     `epochs` times to run optimization
            batch_size : int
                         batch size for SGD-based optimizer
            shuffle : bool
                      whether should shuffle dataset after each epoch
            valids : np.ndarray or list / tuple
                     validation dataset for evaluation
            graph : tf.Graph / None
            config : tf.ProtoConfig / None
            checkpoints : string
                          checkpoints for restoring / saving parameters
            logs : string
                   logs directory for tensorboard
            save : string, `all` / `max` / `min`
                   saving parameter policy.
                   `all` : save each iteration of all epochs
                   `max` : save only max loss / accuracy iteration
                   `min` : save only min loss / accuracy iteration
                   (NOTE: loss that is been saved will be shown in green,
                          while loss that is ignored will be shonw in red)
    """
    ans = session(graph=graph,
                   config=config,
                   checkpoints=checkpoints,
                   logs=logs)
    sess = ans['session']
    saver = ans.get('saver', None)
    summarize = ans.get('summarize', None)
    writer = ans.get('writer', None)
    if y is not None:
        data = [x, y]
    else:
        data = x
    generator, nsamples, iterations = dbs.make_generator(data, batch_size, shuffle, nclass)
    progressor = line(range(nsamples), timeit=True, nprompts=20, enum=True)
    trainop = {'optimizer':optimizer, 'loss':loss}
    saverop = _loss_filter(save)
    if summarize is not None:
        trainop['summarize'] = summarize
    best_result = None
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch+1))
        progress = progressor()
        for (idx, iteration) in progress:
            samples, step = next(generator)
            labels = None
            if isinstance(samples, (list, tuple)):
                samples, labels = samples
            if labels is not None:
                rdict = sess.run(trainop,
                                 feed_dict={xtensor:samples,
                                            ytensor:labels
                                           })
            else:
                rdict = sess.run(trainop,
                                 feed_dict={xtensor:samples})
            if writer is not None:
                writer.add_summary(rdict['summarize'], global_step=epoch)
            _loss = rdict['loss']
            need_save = saverop(_loss, best_result)
            if (idx+1) < iterations:
                if need_save:
                    progress.send(' -- loss: {}{:.6}{}'
                                  .format(colors.fg.green,
                                          _loss,
                                          colors.reset))
                    best_result = _loss
                else:
                    progress.send(' -- loss: {}{:.6}{}'
                                  .format(colors.fg.red,
                                          _loss,
                                          colors.reset))
            if need_save and saver is not None:
                helpers.save(sess, checkpoints, saver, write_meta_graph=False, verbose=False)
        if valids is not None:
            validates(valids, batch_size)
    if writer is not None:
        writer.close()
    sess.close()


def train(x, xtensor, optimizer, loss,
          y=None,
          ytensor=None,
          nclass=None,
          metrics=None,
          epochs=1000,
          batch_size=32,
          shuffle=True,
          valids=None,
          graph=None,
          config=None,
          checkpoints=None,
          logs=None,
          save='all'):
    optimizer = ops.optimizers.get(optimizer)
    loss = ops.losses.get(loss)
    metrics = ops.metrics.get(metrics)
    run(x, xtensor, optimizer, loss, y, ytensor, nclass, metrics, epochs,
        batch_size, shuffle, valids, graph, config, checkpoints, logs, save)