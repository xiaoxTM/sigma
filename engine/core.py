from .. import helpers, colors, dbs, ops
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


def predict_op(input_shape,
               predop=None,
               axis=None,
               dtype=ops.core.int64,
               reuse=False,
               name=None,
               scope=None):
    ops_scope, name = ops.helper.assign_scope(name, scope, 'prediction', reuse)
    if predop is None:
        def _linear(x, axis, name):
            return x
        predop = _linear
    elif isinstance(predop, str):
        if predop == 'argmax':
            predop = ops.core.argmax
        elif predop == 'argmin':
            predop = ops.core.argmin
        else:
            raise ValueError('`predop` must be one of `argmax` or `argmin`.'
                             ' given {}'.format(predop))
    elif not callable(predop):
        raise TypeError('`predop` must be type of None or str or callable.'
                        ' given {}'.format(type(predop)))
    if axis is None:
        axis = 0
    elif axis < 0:
        axis = ops.helper.normalize_axes(input_shape, axis)
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
    def _predict_op(x):
        with ops_scope:
            return predop(x, axis, name=name)
    return _predict_op


def predict(session, x, xtensor, ypred,
            predop=None,
            batch_size=32,
            axis=None,
            dtype=ops.core.int64,
            nclass=None,
            checkpoints=None,
            savedir=None,
            reuse=False,
            name=None,
            scope=None):
    predop = predict_op(ops.core.shape(ypred),
                        predop,
                        axis,
                        dtype,
                        reuse,
                        name,
                        scope)
    ypred = predop(ypred)
    if nclass is None:
        nclass = ops.helper.depth(ypred)
    if checkpoints is not None:
        sess, saver = helpers.load(sess, checkpoints, verbose=True)
    batch_size = min(batch_size, len(x))
    generator, nsamples, iterations = dbs.images.make_generator(x, None,
                                                                xtensor,
                                                                batch_size,
                                                                False,
                                                                nclass)
    progress = line(range(nsamples), timeit=True, nprompts=20, enum=True)
    preds = []
    for (idx, iteration) in progress():
        samples, step = next(generator)
        _ypred = session.run(ypred, feed_dict=samples)
        if savedir is None:
            preds.append(_ypred)
        else:
            os.makedirs(savedir, exist_ok=True)
            for i, images in enumerate(zip(samples, _ypred.astype(np.int8))):
                dbs.images.save_image('{}/{}.png'.format(idx, i),
                                      helpers.stack(images, interval=10,
                                                    value=[0, 0, 0]))
    return preds[0] if len(preds) == 1 else preds


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
        Attibutes
        =========
            iterable : list / tuple / np.ndarray / range
                       iterable object or function that can yield something
            brief : bool
                    if true, show extra information in form of
                        `[current-iter / total-iter]`
            nprompts : int
                       how many prompt (`nc`/`cc`) to mimic the progress
            epochs : int or None
                     total iterations
            multiplier : float
                         multiplier factor for each step
                         the greater the multiplier is,
                         the more progress one prompt represents
            enum : bool
                   return iteration index if true
                   return yield object ONLY else
            timeit : bool
                     time each iteration or not
            show_total_time : bool
                              show the sum of each iteration time if true
            accuracy : int
                       specifying print format precision for time
            message : string or None
                      message to be shown at the begining of each line
            nc : char
                 character to show as running prompt
            cc : char
                 character to show as running and ran prompt for each iteration
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
            if brief:
                if timeit:
                    message = spec.format(prompt, percentage, elapsed, ret)
                else:
                    message = spec.format(prompt, percentage, ret)
            else:
                if timeit:
                    message = spec.format(prompt, idx, epochs, percentage,
                                          elapsed, ret)
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
        save='all',
        emc=None):
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
            emc : dict
                  email configuration, including:
                  - epm : epoch per message
                  - other parameters see @helpers.mail.sendmail
    """
    ans = session(graph=graph,
                  config=config,
                  checkpoints=checkpoints,
                  logs=logs)
    sess = ans['session']
    saver = ans.get('saver', None)
    summarize = ans.get('summarize', None)
    writer = ans.get('writer', None)
    generator, nsamples, iterations = dbs.images.make_generator(x, y,
                                                                xtensor,
                                                                ytensor,
                                                                batch_size,
                                                                shuffle,
                                                                nclass)
    progressor = line(range(nsamples), timeit=True, nprompts=20, enum=True)
    trainop = {'optimizer':optimizer, 'loss':loss}
    saverop = _loss_filter(save)
    if summarize is not None:
        trainop['summarize'] = summarize
    best_result = None
    epm = -1
    if emc is not None:
        emc.get('epm', -1)
        if 'epm' in emc.keys():
            emc.pop('epm')
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch+1))
        progress = progressor()
        for (idx, iteration) in progress:
            samples, step = next(generator)
            rdict = sess.run(trainop,
                             feed_dict=samples)
            if writer is not None:
                writer.add_summary(rdict['summarize'], global_step=epoch)
            _loss = rdict['loss']
            need_save = saverop(_loss, best_result)
            if (idx+1) < iterations: # if generator not stop iterate
                if need_save:
                    best_result = _loss
                    progress.send(' -- loss: {}{:.6}{} / {}{:.6}{}'
                                  .format(colors.fg.green, _loss, colors.reset,
                                          colors.fg.green, best_result,
                                          colors.reset))
                else:
                    progress.send(' -- loss: {}{:.6}{} / {}{:.6}{}'
                                  .format(colors.fg.red, _loss, colors.reset,
                                          colors.fg.green, best_result,
                                          colors.reset))
            if need_save and saver is not None:
                helpers.save(sess, checkpoints, saver,
                             write_meta_graph=False,
                             verbose=False)
        #if valids is not None:
        #    validates(valids, batch_size)
        if epm > 0 and (epoch + 1) % epm == 0:
            helpers.sendmail(emc)
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
    #metrics = ops.metrics.get(metrics)
    run(x, xtensor, optimizer, loss, y, ytensor, nclass, metrics, epochs,
        batch_size, shuffle, valids, graph, config, checkpoints, logs, save)
