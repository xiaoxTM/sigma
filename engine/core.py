from .. import helpers, colors, dbs, ops, layers
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import math
import sigma

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
         epochs,
         #batch_size,
         brief=False,
         nprompts=10,
         timeit=False,
         show_total_time=False,
         accuracy=5,
         message=None,
         nc='x',
         cc='+'):
    """ Python Generator Tips [next|send / yield]
        show line message in a programmatic interactive mode
        to interactivate with line, call generator in the following order:
        (global_idx, elem, epoch, iteration) = (generator)
        while True:
            # do some work here
            (global_idx, elem, epoch, iteration) = generator.send(value)
            if need_break:
                break
        That is, to interactivate next / send with yield in generator, do
        it in the following loop:
                next(generator) \
                                 V
                    -------------|
                    |            |
                    |            V
                  send         yield
                    ^            |
                    |            |
                    |____________|
                   
        Attibutes
        =========
            iterable : list / tuple / np.ndarray / range
                       iterable object or function that can yield something
                       generally speaking, iterable have length of:
                       "iterations * epochs"
            brief : bool
                    if true, show extra information in form of
                        `[current-iter / total-iter]`
            nprompts : int
                       how many prompt (`nc`/`cc`) to mimic the progress
            epochs : int or None
                     total iterations
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
        Returns
        ==========
            tuple of (global_index, iterable_element, epoch, iteration), where:
            global_index : int
                            global index for iterable
            epoch : int
            iteration : int
                        iteration inside epoch
    """
    if epochs is None:
        epochs = 1
    try:
        iterable_size = len(iterable)
    except TypeError:
        raise TypeError('getting the length of `iterable` failed')
    epochsize = intsize(epochs)
    if message is None:
        message = '@'
    nc = nc.encode('utf8')
    cc = cc.encode('utf8')
    # number of iterations per each epoch
    iterations = int(iterable_size / epochs)
    #epochlen = iterations * batch_size
    # rotation times per one `+` / `x`
    scale = nprompts / iterations
    step = max(1, int(scale))
    _prompt = np.asarray([nc] * nprompts, dtype=np.string_)
    itersize = intsize(iterations)
    beg = None
    elapsed = None
    if brief:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%] -- {{:.{}}}(s) {{}}' \
                   .format(epochsize, message, epochs, nprompts, accuracy)
        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%] {{}} '.format(epochsize,
                                                                   message,
                                                                   epochs,
                                                                   message,
                                                                   nprompts)
    else:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' -- {{:.{}}}(s) {{}}'.format(epochsize,
                                                 message,
                                                 epochs,
                                                 nprompts,
                                                 itersize,
                                                 itersize,
                                                 accuracy)
        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%] {{}} ' \
                   .format(epochsize, message, epochs, itersize, itersize, nprompts)
    def _line():
        totaltime = 0
        prev = 0
        idx = 0
        while True:
            elem = iterable[idx % iterable_size]
            time_beg = timer()
            epoch = int(idx / iterations)
            iteration = (idx % iterations)
            ret = (yield (idx, elem, epoch, iteration))
            #if ret is None:
            #    ret = ''
            # convert iteration index to nprompts index
            block_beg = int(iteration * scale)
            if block_beg >= nprompts:
                block_beg = nprompts - step
            block_end = block_beg + step
            elapsed = (timer() - time_beg)
            totaltime += elapsed
            iteration += 1

            percentage = int(iteration * 100 / iterations)

            if _prompt[block_beg] == nc or prev != block_beg or percentage == 100:
                _prompt[block_beg:block_end] = cc
                if prev != block_beg:
                    _prompt[prev:block_beg] = cc
            else:
                _prompt[block_beg:block_end] = nc
            prev = block_beg
            prompt = _prompt[:block_end].tostring().decode('utf-8')
            if brief:
                if timeit:
                    message = spec.format(epoch+1, prompt, percentage,
                                          elapsed, ret)
                else:
                    message = spec.format(epoch+1, prompt, percentage, ret)
            else:
                if timeit:
                    message = spec.format(epoch+1, prompt, iteration, iterations,
                                          percentage, elapsed, ret)
                else:
                    message = spec.format(epoch+1, prompt, iteration, iterations,
                                          percentage, ret)
            end_flag = ''
            if percentage == 100:
                # start new epoch
                end_flag = '\n'
            print(message, end=end_flag, flush=True)
            idx += 1
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
    # make dataset generator
    generator, nsamples, iterations = dbs.images.make_generator(x, y,
                                                                xtensor,
                                                                ytensor,
                                                                batch_size,
                                                                shuffle,
                                                                nclass)
    total_iterations = iterations * epochs
    progressor = line(range(total_iterations),
                      epochs=epochs,
                      #batch_size=batch_size,
                      timeit=True,
                      nprompts=20)()
    # tensors to be calculated
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
    (global_idx, _, epoch, _) = next(progressor)
    while True:
        if epoch >= epochs: # if generator not stop iterate
            break
        samples, step = next(generator)
        rdict = sess.run(trainop,
                         feed_dict=samples)
        if writer is not None:
            writer.add_summary(rdict['summarize'], global_step=global_idx)
        _loss = rdict['loss']
        need_save = saverop(_loss, best_result)
        if need_save:
            best_result = _loss
            (global_idx, _, epoch, _) = progressor.send(
                ' -- loss: {}{:.6}{} / {}{:.6}{}$'.format(
                    colors.fg.green, _loss, colors.reset,
                    colors.fg.green, best_result, colors.reset))
        else:
            (global_idx, _, epoch, _) = progressor.send(
                ' -- loss: {}{:.6}{} / {}{:.6}{}$'.format(
                    colors.fg.red, _loss, colors.reset,
                    colors.fg.green, best_result, colors.reset))

        if need_save and saver is not None:
            helpers.save(sess, checkpoints, saver,
                         write_meta_graph=False,
                         verbose=False)
        #if valids is not None:
        #    validates(valids, batch_size)
        #if epm > 0 and (epoch + 1) % epm == 0:
        #    helpers.sendmail(emc)
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


def build(input_shape, build_func, loss,
          dtype='float32',
          name=None,
          reuse=False,
          scope=None, **kwargs):
    """ build network architecture
        Attributes
        ==========
        input_shape : list / tuple
                      input shape for network entrance
        build_func : callable
                     callable function receives only one
                     parameter. should have signature of:
                     `def build_func(x) --> tensor:`
        loss : string or callable
               loss to be maximized or mimimized
        dtype : string
                data type of input layer
        name : string
               name of input layer
        reuse : bool
        scope : string
        **kwargs : parameters passed to loss function (layer)
        
        Returns
        ==========
        inputs : tensor
                 input tensor to be feed by samples
        loss : loss
    """
    with sigma.defaults(reuse=reuse, scope=scope):
        inputs = layers.base.input_spec(input_shape, dtype=dtype, name=name)
        x = build_func(inputs)
        return inputs, layers.losses.get(loss, x, **kwargs)