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
    progressor = helper.line(iterable=None,
                             epochs=None,
                             iterations=iterations,
                             timeit=True)[0]()
    preds = []
    sigma.status.is_training = False
    (global_idx, _, epoch, iteration) = next(progressor)
    while epoch < 1:
        samples, step = next(generator)
        pred = ops.core.run(session, ypred, feed_dict=samples)
        if savedir is None:
            preds.append(pred)
        else:
            os.makedirs(savedir, exist_ok=True)
            for i, images in enumerate(zip(samples, pred.astype(np.int8))):
                dbs.images.save_image('{}/{}.png'.format(global_idx, i),
                                      helpers.stack(images, interval=10,
                                                    value=[0, 0, 0]))
        (global_idx, _, epoch, iteration) = next(progressor)
    sigma.status.is_training = True
    if len(preds) > 0:
        return preds[0] if len(preds) == 1 else preds


def validate(session, validop, generator, iterations):
    acc = None
    if 'metric' in validop.keys():
        acc = 0
    loss = 0
    sigma.status.is_training = False
    for iteration in range(iterations):
        samples, step = next(generator)
        rdict = ops.core.run(session, validop, feed_dict=samples)
        accuracy = rdict.get('metric', None)
        if accuracy is not None:
            if isinstance(accuracy, (list, tuple)):
                # if metric are built from metrics.*
                # it will include
                # [accuracy, update_op]
                accuracy = accuracy[0]
            acc += accuracy
        loss += rdict['loss']
    if acc is not None:
        acc = acc / iterations
    loss = loss / iterations
    sigma.status.is_training = True
    return (loss, acc)


def _save_op(savemode='all', modetarget='loss'):
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

    if savemode == 'all':
        return _all
    elif savemode == 'max':
        return _max
    elif savemode == 'min':
        return _min
    else:
        raise ValueError('`savemode` must be `all`, `max` or `min`. given {}'
                         .format(savemode))


def session(target='',
            graph=None,
            config=None,
            initializers=None,
            checkpoints=None,
            logs=None,
            verbose=True):
    """ get session and setup Graph, GPU, checkpoints, logs
    """
    sess = ops.core.session(target, graph, config, initializers)
    ans = {'session': sess}
    if checkpoints is not None:
        sess, saver = helpers.load(sess, checkpoints, verbose=verbose)
        ans['session'] = sess
        ans['saver'] = saver
    if logs is not None:
        summarize = ops.core.summary_merge()
        writer = ops.core.summary_writer(logs, sess.graph)
        ans['summarize'] = summarize
        ans['writer'] = writer
    return ans


def run(x, xtensor, optimizer, loss,
        metric=None,
        y=None,
        ytensor=None,
        nclass=None,
        epochs=1000,
        batch_size=32,
        shuffle=True,
        valids=None,
        graph=None,
        config=None,
        checkpoints=None,
        logs=None,
        savemode='all',
        modetarget='loss',
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
            loss : Tensor
                   objectives network tries to minimize / maximize
            metrics : Tensor / None
                      measurement of result accuracy
            y : np.ndarray or list / tuple / None
                ground truth for x.
            ytensor : tf.placeholder / None
                      y placeholder of the network
            nclass : int / None
                     number of class for classification
                     Note, None indicates no one_hot op when generate
                     (sample, label) pairs
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
            savemode : string, `all` / `max` / `min`
                       saving parameter policy.
                       `all` : save each iteration of all epochs
                       `max` : save only max loss / accuracy iteration
                       `min` : save only min loss / accuracy iteration
            modetarget : string, `loss` / `metric`
                         which target to decide save or not
                         NOTE that when modetarget is metric and metric
                         is not given, save modetarget will change to
                         loss automatically
            emc : dict
                  email configuration, including:
                  - epm : epoch per message
                  - other parameters see @helpers.mail.sendmail
    """
    # //FIXME: remove validating time from final iteration of train time
    if modetarget not in ['loss', 'metric']:
        raise ValueError('modetarget must be `loss` or `metric`. given {}'
                         .format(colors.red(modetarget)))
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
    if valids is not None:
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
    validop = {'loss': loss}
    def _encode_loss_and_acc(loss, acc):
        return '{} / {}'.format(colors.blue(round(loss, 6), '{:0<.6f}'),
                                colors.green(round(acc, 6), '{: >.6f}'))
    def _encode_loss(loss, acc):
        return '{}'.format(colors.blue(round(loss, 6), '{:<.6}'))
    def _get_acc(result):
        return result['metric'][0]
    get_acc = lambda x: None
    encodeop = _encode_loss
    if metric is not None:
        trainop['metric'] = metric
        validop['metric'] = metric
        encodeop = _encode_loss_and_acc
        get_acc = _get_acc
    if summarize is not None:
        trainop['summarize'] = summarize
    saverop = _save_op(savemode, modetarget)
    best_result = None
    epm = -1
    if emc is not None:
        emc.get('epm', -1)
        if 'epm' in emc.keys():
            emc.pop('epm')
    (global_idx, _, epoch, iteration) = next(progressor)
    while epoch < epochs:
        validmessage = ''
        samples, step = next(generator)
        rdict = ops.core.run(sess, trainop, feed_dict=samples)
        if writer is not None:
            ops.core.add_summary(writer, rdict['summarize'],
                                 global_step=global_idx)
        trainloss = rdict['loss']
        trainacc = get_acc(rdict)

        # begin of evaluation
        if (iteration + 1) == iterations and \
          valid_gen is not None:
            validloss, validacc = validate(sess, validop, valid_gen, valid_iters)
            validmessage = ' => {}'.format(encodeop(validloss, validacc))
        # end of evaluation
        current = trainloss
        if modetarget == 'metric' and metric is not None:
            current = trainacc
        if saverop(current, best_result):
            # if current loss is better than best result
            # save it to best_result
            best_result = current
            if saver is not None:
                helpers.save(sess, checkpoints, saver,
                             write_meta_graph=False,
                             verbose=False)
#        print('loss:', trainloss, '- acc:', trainacc)
        trainmessage = encodeop(trainloss, trainacc)
        (global_idx, _, epoch, iteration) = progressor.send(
            '{{{}{}}}'.format(trainmessage,
                              validmessage))
        if epm > 0 and (epoch + 1) % epm == 0:
            helpers.sendmail(emc)
    ops.core.close_summary_writer(writer)
    ops.core.close_session(sess)

def train(x, xtensor, optimizer, loss,
          metric=None,
          y=None,
          ytensor=None,
          nclass=None,
          epochs=1000,
          batch_size=32,
          shuffle=True,
          valids=None,
          graph=None,
          config=None,
          checkpoints=None,
          logs=None,
          savemode='all',
          modetarget='loss'):
    loss_op = ops.losses.get(loss)
    optimization_op = ops.optimizers.get(optimizer).minimize(loss_op)
    metric_op = ops.metrics.get(metric)
    sigma.status.is_training = True
    run(x, xtensor,
        optimization_op,
        loss_op,
        metric_op,
        y, ytensor,
        nclass,
        epochs,
        batch_size,
        shuffle,
        valids,
        graph,
        config,
        checkpoints,
        logs,
        savemode,
        modetarget)
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
            labels = layers.base.label_spec(label_shape,
                                            dtype=ydtype,
                                            name=yname)
            # build_fun should returns `loss`, `metrics`
            # that is, `x` in the form of:
            #    [loss, metric]
            # or (loss, metric)
            # or {'loss': loss, 'metric': metric}
            x = build_fun(inputs, labels, **kwargs)
        else:
            labels = None
            x = build_fun(inputs, **kwargs)
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
