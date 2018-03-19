from .. import helpers, colors, dbs, ops, layers
import tensorflow as tf
import sigma

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
    sigma.status.is_training = False
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
    sigma.status.is_training = True
    return preds[0] if len(preds) == 1 else preds


def validate(session, generator, metric, iterations):
    acc = 0
    validop = {'acc':metric}
    sigma.status.is_training = False
    for iteration in range(iterations):
        samples, step = next(generator)
        rdict = session.run(validop, feed_dict=samples)
        accuracy = rdict['acc']
        if isinstance(accuracy, (list, tuple)):
            # if metric are built from tf.metrics.*
            # it will include
            # [accuracy, update_op]
            accuracy = accuracy[0]
        acc += accuracy
    mean = acc / iterations
    sigma.status.is_training = True
    return mean


def session(target='',
            graph=None,
            config=None,
            initializers=None,
            checkpoints=None,
            logs=None,
            verbose=True):
    sess = tf.Session(target, graph, config)
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
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


def _save_op(save='all'):
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
        epochs=1000,
        batch_size=32,
        shuffle=True,
        valids=None,
        metric=None,
        graph=None,
        config=None,
        checkpoints=None,
        logs=None,
        savemode='all',
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
    # //FIXME: remove validating time from final iteration of train time
    # //FUTURE: to show accuracy after each iteration of training
    # //FUTURE: `savemode` with respect to loss or accuracy
    # //FUTURE: to show progress <epoch @epochs>[+++++++++x, iter/iters, p%]{loss / acc => loss / acc}@time$
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
    valid_gen = None
    valid_iters = 0
    if valids is not None and metric is not None:
        if isinstance(valids, dict):
            vx = valids['x']
            vy = valids['y']
        elif isinstance(valids, (list, tuple)):
            vx, vy = valids
        else:
            raise TypeError('`valids` for run must have type of '
                            '{}. given {}'.format(
                                colors.blue(
                                    'dict / list / tuple'),
                                colors.red(type(valids))))
        valid_gen, _, valid_iters = dbs.images.make_generator(vx, vy,
                                                              xtensor,
                                                              ytensor,
                                                              batch_size,
                                                              False,
                                                              nclass)
    progressor = helpers.line(None,
                              epochs=epochs,
                              iterations=iterations,
                              feedbacks=True, # use `send` to get next data instead of `next`
                              timeit=True,
                              nprompts=10)[0]()
    # tensors to be calculated
    trainop = {'optimizer':optimizer, 'loss':loss}
    saverop = _save_op(savemode)
    if summarize is not None:
        trainop['summarize'] = summarize
    best_result = None
    epm = -1
    if emc is not None:
        emc.get('epm', -1)
        if 'epm' in emc.keys():
            emc.pop('epm')
    (global_idx, _, epoch, iteration) = next(progressor)
    while epoch < epochs:
        acc = '$'
        samples, step = next(generator)
        rdict = sess.run(trainop,
                         feed_dict=samples)
        if writer is not None:
            writer.add_summary(rdict['summarize'], global_step=global_idx)
        _loss = rdict['loss']

        # begin of evaluation
        if (iteration + 1) == iterations and \
          valid_gen is not None:
            acc = validate(sess, valid_gen, metric, valid_iters)
            acc = ' -- acc: {}$'.format(colors.blue(acc, '{:.6}'))
        # end of evaluation

        if saverop(_loss, best_result):
            # if current loss is better than best result
            # save it to best_result
            best_result = _loss
            loss_string = colors.green(_loss, '{:6}')
            if saver is not None:
                helpers.save(sess, checkpoints, saver,
                             write_meta_graph=False,
                             verbose=False)
        else:
            loss_string = colors.red(_loss, '{:6}')

        best_result_string = colors.green(best_result, '{:6}')
        (global_idx, _, epoch, iteration) = progressor.send(
            ' -- loss: {} / {}{}'.format(loss_string,
                                         best_result_string,
                                         acc))
        if epm > 0 and (epoch + 1) % epm == 0:
            helpers.sendmail(emc)
    if writer is not None:
        writer.close()
    sess.close()


def train(x, xtensor, optimizer, loss,
          y=None,
          ytensor=None,
          nclass=None,
          epochs=1000,
          batch_size=32,
          shuffle=True,
          valids=None,
          metric=None,
          graph=None,
          config=None,
          checkpoints=None,
          logs=None,
          savemode='all'):
    sigma.status.is_training = True
    loss_op = ops.losses.get(loss)
    optimization_op = ops.optimizers.get(optimizer).minimize(loss_op)
    metric_op = ops.metrics.get(metric)
    run(x, xtensor,
        optimization_op,
        loss_op,
        y, ytensor,
        nclass,
        epochs,
        batch_size,
        shuffle,
        valids,
        metric_op,
        graph,
        config,
        checkpoints,
        logs,
        savemode)
    sigma.status.is_training = False


def build(input_shape,
          build_fun,
          label_shape=None,
          dtype='float32',
          name=None,
          reuse=False,
          scope=None,
          **kwargs):
    """ build network architecture
        Attributes
        ==========
        input_shape : list / tuple
                      input shape for network entrance
        label_shape : list / tuple
                      label shape for network entrance
        build_fun : callable
                    callable function receives only one
                    parameter. should have signature of:
                    `def build_fun(x) --> (tensor, tensor):`
                    where the first tensor is loss and the
                    second tensor is metric (can be None)
        dtype : string / list / dict
                data type of input / label layer
        name : string / list / dict
               name of input / label layer
        reuse : bool
        scope : string
        kwargs : None or dict
                 parameters passed to build_fun
                 e.g.,
                     loss='margin_loss',
                     metric='accuracy',
                     fastmode=True,
                     ...
                     etc.
        Returns
        ==========
            ([inputs, labels], [loss, metric])
            inputs : tensor
                     input tensor to be feed by samples
            labels : tensor
                     placeholder for ground truth
            loss   : tensor
                     loss to be optimized
            metric : tensor
                     metric to measure the performance
    """
    if isinstance(dtype, str):
        xdtype, ydtype = dtype, dtype
    elif isinstance(dtype, list):
        if len(dtype) == 1: # e.g., ['float32']
            xdtype, ydtype = dtype[0], dtype[0]
        elif len(dtype) == 2: # e.g., ['float32', int32]
            xdtype, ydtype = dtype
        else:
            raise ValueError('`dtype` as list must has length of 1 or 2. '
                             'given {}'.format(len(dtype)))
    elif isinstance(dtype, dict):
        xdtype = dtype['inputs']
        ydtype = dtype['labels']
    else:
        raise TypeError('`dtype` must be str / list / dict. given {}'
                        .format(type(dtype)))
    if  name is None:
        xname, yname = name, name
    elif isinstance(name, str):
        xname, yname = 'input-{}'.format(name), 'label-{}'.format(name)
    elif isinstance(name, list):
        if len(name) == 1:
            xname, yname = name[0], name[0]
        elif len(name) == 2:
            xname, yname = name
        else:
            raise ValueError('`name` as list must has length of 1 or 2. '
                             'given {}'.format(len(name)))
    elif isinstance(name, dict):
        xname = name['inputs']
        yname = name['labels']
    else:
        raise TypeError('`name` must be str / list / dict. given {}'
                        .format(type(name)))

    with sigma.defaults(reuse=reuse, scope=scope):
        inputs = layers.base.input_spec(input_shape, dtype=xdtype, name=xname)
        if label_shape is not None:
            labels = layers.base.label_spec(label_shape, dtype=ydtype, name=yname)
            # build_fun should returns `loss`, `metrics`
            # that is, `x` in the form of:
            #    [loss, metric]
            # or (loss, metric)
            # or {'loss': loss, 'metric': metric}
            x = build_fun(inputs, labels, **kwargs)
        else:
            labels = None
            x = buuild_fun(inputs, **kwargs)
        if ops.helper.is_tensor(x):
            loss, metric = x, None
        elif isinstance(x, (list, tuple)):
            if len(x) == 1:
                loss, metric = x, None
            elif len(x) == 2:
                loss, metric = x
            else:
                raise ValueError('The return value of `build_fun` must have'
                                 ' length of 1 or 2 in list / tuple. given {}'
                                 .format(len(x)))
        elif isinstance(x, dict):
            loss = x['loss']
            metric = x.get('metric', None)
        else:
            raise TypeError('The return value type of `build_fun` must be'
                            ' tensor / list / tuple / dict. given {}'
                            .format(type(x)))
        return ([inputs, labels], [loss, metric])
