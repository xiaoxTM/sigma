from .. import helpers, colors, dbs, ops, layers
import sigma
import os
import os.path
import argparse
import numpy as np


def phase(mode='train'):
    if mode not in ['train', 'predict']:
        raise ValueError('`phase` mode only support `train/predict`. given {}'
                         .format(mode))
    def _phase(fun):
        def _wrap(*args, **kwargs):
            if mode == 'train':
                sigma.status.is_training = True
            else:
                sigma.status.is_training = False
            x = fun(*args, **kwargs)
            sigma.status.is_training = not sigma.status.is_training
            return x
        return _wrap
    return _phase


def predict_op(input_shape,
               predop=None,
               axis=None,
               dtype=ops.core.int64,
               reuse=False,
               name=None,
               scope=None):
    ops_scope, ltype, name = ops.helper.assign_scope(name, scope, 'prediction', reuse)
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


@phase('predict')
def predict(sess,
            y, # tensor
            generator,
            iterations,
            x=None, # numpy.ndarray
            ground_truth=None,
            predop='argmax',
            batch_size=32,
            axis=None,
            dtype=ops.core.int64,
            nclass=None,
            checkpoint=None,
            savedir=None,
            reuse=False,
            name=None,
            scope=None):
    if isinstance(y, tuple):
        input_shape = ops.core.shape(y)
        ypred = predict_op(input_shape,
                           predop,
                           axis,
                           dtype,
                           reuse,
                           name,
                           scope)(y[0])
        y = (ypred, y[1:])
        if nclass is None:
            nclass = ops.helper.depth(y[0])
    else:
        input_shape = ops.core.shape(y)
        y = predict_op(input_shape,
                       predop,
                       axis,
                       dtype,
                       reuse,
                       name,
                       scope)(y)
        if nclass is None:
            nclass = ops.helper.depth(y)
    if checkpoint is not None:
        sess, saver = helpers.load(sess, checkpoint, verbose=True)
    if generator is None and iterations is None:
        if x is None:
            ValueError("generator, iterations and x cannot ALL be None")
        batch_size = min(batch_size, len(x))
        generator, _, iterations = dbs.images.make_generator(x,
                                                             ground_truth,
                                                             batch_size,
                                                             False,
                                                             nclass)
    progressor = helpers.line(iterable=None,
                             epochs=None,
                             iterations=iterations,
                             timeit=True)[0]()
    preds = []
    (global_idx, _, epoch, iteration) = next(progressor)
    while epoch < 1:
        samples, step = next(generator)
        pred = ops.core.run(sess, y, feed_dict=samples)
        if savedir is None:
            preds.append([pred])
        else:
            os.makedirs(savedir, exist_ok=True)
            for i, images in enumerate(pred.astype(np.int8)):
                dbs.images.save_image('{}/{}.png'.format(global_idx, i), images)
        (global_idx, _, epoch, iteration) = next(progressor)
    if len(preds) > 0:
        return preds[0] if len(preds) == 1 else preds


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


def checkpoint_save(checkpoint, saver, **kwargs):
    if saver is None:
        return lambda x:x
    def _checkpoint_save(session):
        helpers.save(session, checkpoint, saver, **kwargs)
    return _checkpoint_save


def log_summary(writer):
    if writer is None:
        return lambda summary, global_step=None:summary
    def _log_summarize(summary, global_step=None):
        ops.core.add_summary(writer, summary, global_step=global_step)
    return _log_summarize


def session(target='',
            graph=None,
            config=None,
            initializers=None,
            checkpoint=None,
            log=None,
            debug=False,
            address=None,
            verbose=True):
    """ get session and setup Graph, GPU, checkpoints, logs
    """
    sess = ops.core.session(target, graph, config, initializers, debug, address)
    saver, summarize, writer = None, None, None
    if checkpoint is not None:
        parent_dir = checkpoint
        if checkpoint[-1] != '/':
            parent_dir = checkpoint.rsplit('/', 1)[0]
        os.makedirs(parent_dir, exist_ok=True)
        sess, saver = helpers.load(sess, checkpoint, saver, verbose=verbose)
    if log is not None:
        summarize = ops.core.summary_merge()
        # in case none tf.summary.* are invoked
        if summarize is not None:
            writer = ops.core.summary_writer(log, sess.graph)
    return (sess, saver, summarize, writer)


def run(session,
        train_op,
        generator,
        iterations,
        encode_op,
        checkpoint_save_op,
        summarize_op,
        valid_op=None,
        epochs=1000,
        filename=None,
        save_mode='all',
        save_target='loss',
        emc=None):
    """ run to train networks
        Attributes
        ==========
            trainop : dict
                      operations to be run to train networks
                      for example:
                          trainop = {'loss':loss, 'optimizer':optimizer}
            generator : generator
                        generate samples for training networks
            iterations : int
                         total iterations for each epoch for training networks
            validation : callable
                         function for validating networks
                         should have signature of
                         `
                            def validation(session):
                                # tons of thousand of codes here
                                return {'loss': loss, 'metric': metric}
                         `
            epochs : int
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
    # FIXME: remove validating time from final iteration of train time
    if save_target not in ['loss', 'metric']:
        raise ValueError('save_target must be `loss` or `metric`. given {}'
                         .format(colors.red(save_target)))
    progressor = helpers.line(None,
                              epochs=epochs,
                              iterations=iterations,
                              brief=False,
                              feedbacks=True,
                              timeit=True,
                              nprompts=20)[0]()
    saver_op = _save_op(save_mode, save_target)
    best_result = None
    epm = -1
    if emc is not None:
        emc.get('epm', -1)
        if 'epm' in emc.keys():
            emc.pop('epm')
    (global_idx, _, epoch, iteration) = next(progressor)
    records = []
    while epoch < epochs:
        validmessage = ''
        samples, step = next(generator)
        rdict = train_op(samples)
        summarize_op(rdict.get('summarize', None), global_step=global_idx)
        trainloss = rdict['loss']
        trainacc = rdict.get('metric', None)

        # begin of evaluation
        if (iteration + 1) == iterations:
            record = [trainloss]
            if trainacc is not None:
                record += [trainacc]
            if valid_op is not None:
                validloss, validacc = valid_op(global_idx)
                validmessage = ' => {}'.format(encode_op(validloss, validacc))
                record += [validloss]
                if validacc is not None:
                    record += [validacc]
            records.append(record)
        # end of evaluation
        current = trainloss
        if save_target == 'metric' and trainacc is not None:
            current = trainacc
        if saver_op(current, best_result):
            # if current loss is better than best result
            # save it to best_result
            best_result = current
            checkpoint_save_op(session)
        trainmessage = encode_op(trainloss, trainacc)
        (global_idx, _, epoch, iteration) = progressor.send(
            '{{{}{}}}'.format(trainmessage,
                              validmessage))
        if epm > 0 and (epoch + 1) % epm == 0:
            helpers.sendmail(emc)
    if filename is not None:
        np.savetxt(filename, records)


@phase('train')
def train(sess,
          generator,
          iterations,
          train_op,
          valid_op,
          metric=None,
          valid_gen=None,
          valid_iters=None,
          writer=None,
          summarize=None,
          checkpoint_save_op=None,
          epochs=1000,
          filename=None,
          save_mode='all',
          save_target='loss'):
    """ train networks with samples (may also validate samples)
        Attributes
        ==========
            generator : generator
                        generator to generate sample-label pairs for train
            iterations : total iterations for each epoch for train dataset
            optimizer : str / Optimizer
                        optimizer to minimize / maximize loss
            loss : str / tensor
                   objective loss to be optimized by optimizer
            metric : None / str / tensor
                     metric to measure the performance after each epoch if available
            valid_gen : generator
                        generator to generate sample-label pairs for validate
            valid_iters : int
                          total iterations for validation
            epochs : int
                     epochs to train throughout the train-dataset
            # graph : Graph
            # config : dict
            #          gpu configuration
            # checkpoint : str
            #              checkpoint to store median train result
            # log : str
            #       log directory for tensorboard visualization
            # address : str
            #           address for tfdebug and tensorboard debugging
            #           should in the form of:
            #             hostname:port
            save_mode : str
                       `all` for saving all results to checkpoint
                       `min` for saving minimal value of current and global results
                       `max` for saving maximal value of current and global results
            save_target : string
                         which to save
                         `loss` for saving loss
                         `metric` for saving metric
    """

    summarize_op = log_summary(writer)
    if summarize is not None:
        train_op['summarize'] = summarize

    train_fun = lambda samples:ops.core.run(sess, train_op,
                                            feed_dict=samples)
    valid_fun = None
    if valid_gen is not None and valid_iters is not None:
        @phase('predict')
        def _valid_fun(global_step=None):
            loss = 0
            for iteration in range(iterations):
                samples, step = next(valid_gen)
                ans = ops.core.run(sess, valid_op, feed_dict=samples)
                loss += ans['loss']
            loss = loss / iterations
            return (loss, None)
        valid_fun = _valid_fun
    if metric is not None:
        if isinstance(metric, (list, tuple)):
            # in case of using tf.metrics.*, metrics incudes three parts:
            # metric_measure, metric_update, metric_initializer
            # where metric_measure op returns metric
            #       metric_update op returns updating of metric
            #       metric_initializer op initialize / reset metric_measure
            metric_measure, metric_update, metric_initializer = metric
            train_op['update'] = metric_update
            def _train_fun(samples):
                # initialize / reset metric
                ops.core.run(sess, metric_initializer)
                ans = ops.core.run(sess, train_op, feed_dict=samples)
                acc = ops.core.run(sess, metric_measure)
                ans['metric'] = acc
                return ans
            train_fun = _train_fun
            if valid_fun is not None:
                valid_op['update'] = metric_update
                @phase('predict')
                def _valid_fun(global_step=None):
                    ops.core.run(sess, metric_initializer)
                    loss = 0
                    for iteration in range(iterations):
                        samples, step = next(valid_gen)
                        ans = ops.core.run(sess, valid_op, feed_dict=samples)
                        loss += ans['loss']
                    loss = loss / iterations
                    return (loss, ops.core.run(sess, metric_measure))
                valid_fun = _valid_fun
        elif callable(metric):
            # in case of customized metric
            # metric should have form of
            #    def metric_fun(sess, op, samples) -> (loss, metric)
            def _train_fun(samples):
                return metric(sess, train_op, samples)
            train_fun = _train_fun
            if valid_fun is not None:
                @phase('predict')
                def _valid_fun(global_step=None):
                    loss = 0.0
                    acc  = 0.0
                    for iteration in range(iterations):
                        samples, step = next(valid_gen)
                        ans = metric(sess, train_op, samples)
                        loss += ans['loss']
                        acc  += ans['metric']
                    return (loss / iterations, acc / iterations)
                valid_fun = _valid_fun
    if metric is not None:
        def encode_fun(loss, acc):
            return '{} / {}'.format(colors.blue(round(loss, 6), '{:0<.6f}'),
                                    colors.green(round(acc, 6), '{: >.6f}'))
    else:
        def encode_fun(loss, acc):
            return '{}'.format(colors.blue(round(loss, 6), '{:<.6}'))

    run(sess,
        train_fun,
        generator,
        iterations,
        encode_fun,
        checkpoint_save_op,
        summarize_op,
        valid_fun,
        epochs,
        filename,
        save_mode,
        save_target)


def build_reader(build_fun, **kwargs):
    (input_shape, label_shape), (train, valid, test) = build_fun()
    if isinstance(input_shape, tuple):
        inputs = tuple([layers.base.input_spec(shape,
                  dtype=ops.core.float32,
                  name='inputs-{}'.format(i)) for i,shape in enumerate(input_shape)])
    else:
        inputs = layers.base.input_spec(input_shape,
                                        dtype=ops.core.float32,
                                        name='inputs')
    labels = None
    if label_shape is not None:
        labels =layers.base.label_spec(label_shape,
                                       dtype=ops.core.float32,
                                       name='labels')

    valid_gen, valid_iters = None, None
    test_gen, test_iters = None, None
    if isinstance(train, str):
        train_gen, _, iterations = dbs.images.make_generator_from_list(train, **kwargs)
        kwargs['shuffle'] = False
        if valid[0] is not None and valid[1] is not None:
            valid_gen, _, valid_iters = dbs.images.make_generator_from_list(valid, **kwargs)
            valid_gen = valid_gen(inputs, labels)
        if test is not None:
            test_gen, _, valid_iters = dbs.images.make_generator_from_list(test, **kwargs)
            test_gen = test_gen(inputs, labels)
    elif isinstance(train, (list, tuple)):
        train_gen, _, iterations = dbs.images.make_generator(train[0],
                                                             train[1],
                                                             **kwargs)
        kwargs['shuffle'] = False
        if valid[0] is not None and valid[1] is not None:
            valid_gen, _, valid_iters = dbs.images.make_generator(valid[0],
                                                                  valid[1],
                                                                  **kwargs)
            valid_gen = valid_gen(inputs, labels)
        if test[0] is not None:
            test_gen, _, test_iters = dbs.images.make_generator(test[0],
                                                                test[1],
                                                                **kwargs)
            test_gen = test_gen(inputs, labels)
    train_gen = train_gen(inputs, labels)
    #print("train-gen:", train_gen)
    return (inputs, labels), \
           (train_gen, iterations), \
           (valid_gen, valid_iters), \
           (test_gen, test_iters)


def build_model(inputs,
                build_fun,
                labels=None,
                collections=None,
                summary=None,
                check_output_shape=True,
                check_input_shape=True,
                reuse=False,
                scope=None,
                **kwargs):
    """ build network architecture
        Attributes
        ==========
        inputs : tensor
                 input for network entrance
        build_fun : callable
                    callable function receives only one
                    parameter. should have signature of:
                    `def build_fun(x) --> (tensor, tensor):`
                    where the first tensor is loss and the
                    second tensor is metric (can be None)
        labels : tensor
                 label shape for network entrance
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
    with sigma.defaults(check_input_shape=check_input_shape,
                        check_output_shape=check_output_shape,
                        collections=collections,
                        summary=summary,
                        reuse=reuse,
                        scope=scope):
        x = build_fun(inputs, labels, **kwargs)
    if ops.helper.is_tensor(x):
        inference, loss, metric = None, x, None
    elif isinstance(x, (list, tuple)):
        if len(x) == 1:
            inference, loss, metric = None, x, None
        elif len(x) == 2:
            loss, metric = x
            inference = None
        elif len(x) == 3:
            inference, loss, metric = x
        else:
            raise ValueError('The return value of `build_fun` must have'
                             ' length of 1, 2  or 3 in list / tuple. given {}'
                             .format(len(x)))
    elif isinstance(x, dict):
        loss = x['loss']
        metric = x.get('metric', None)
        inference = x.get('inference', None);
    else:
        raise TypeError('The return value type of `build_fun` must be'
                        ' tensor / list / tuple / dict. given {}'
                        .format(type(x)))
    return [inference, loss, metric]


def build_parser():
    parser = argparse.ArgumentParser()
    # begin of options for training
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--eid', type=str, default=None)
    parser.add_argument('--address', type=str, default=None)
    parser.add_argument('--filename', type=str, default=None)
    #or typically : parser.address='localhost:6064'
    parser.add_argument('--save-mode', type=str, default='all')
    parser.add_argument('--save-target', type=str, default='loss')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--auto-timestamp', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=True)
    # parser.add_argument('--shuffle', type=bool, default=True)
    # end of options for training

    # begin of options for testing
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--axis', type=int, default=None)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--predop', type=str, default='argmax')
    # end of options for testing

    return parser


def build_experiment(build_model_fun,
                     build_reader_fun,
                     optimizer,
                     batch_size=32,
                     shuffle=True,
                     filename=False,
                     collections=None,
                     summary=None,
                     check_output_shape=True,
                     check_input_shape=True,
                     reuse=False,
                     scope=None,
                     model_config=None,
                     generator_config=None,
                     optimizer_config=None,
                     config=None):
    """ build experiment
        this will automatically add timestamp to checkpoints and logs

        Attributes
        ==========
            build_model_fun : callable
                              function to build network structure
                              this will be passed to sigma.build
                              and should have signature of
                              `
                                  def build_network(inputs, labels, **kwargs):
                                      # tons of codes here
                                      return [loss, metric]
                              `
                              see sigma.apps.capsules.[cifar|mnist].build_func
                              for examples
            build_reader_fun : callable
                               function to load dataset
                               this function should return another function
                               that return
                               (input_shape, label_shape), ([x, y], (vx, vy))
                               or
                               (input_shape, label_shape), (train-filelist, valid-filelist)
                               where x is the samples, y is the labels corresponding to x
                               vx is the samples from valid dataset, vy is the labels
                               corresponding to vy
                               should have signature of
                               `
                                   def build_reader(#necessary parameters here):
                                       def _build_reader(**kwargs):
                                           # loading dataset from files here
                                           return (input_shape, label_shape), \
                                                  ([x, y], [vx, vy])
                                       return _build_reader
                               `
                               see sigma.engine.io.[mnist|cifar] for examples
            optimizer : string / sigma.ops.core.Optimizer
                        optimizer to optimize loss and train parameters
            filename : string / bool / None
                       if None, will print no network structure information
                       if False, print to terminal
                       if string, print to file
            model_config : dict
                           parameters passed to build_model_fun
            generator_config : dict
                               parameters passed to build_reader_fun
            optimizer_config : dict
                               parameters passed to ops.optimizer.get
            config : dict:
                         configuration
    """
    if model_config is None:
        model_config = {}
    if generator_config is None:
        generator_config = {}
    generator_config['batch_size'] = batch_size
    generator_config['shuffle'] = shuffle
    if optimizer_config is None:
        optimizer_config = {}

    #----- read the dataset -----#
    (inputs, labels), \
    (generator, iterations), \
    (valid_gen, valid_iters), \
    (test_gen, test_iters) = build_reader(build_reader_fun,
                                            **generator_config)

    #----- build networks -----#
    [inference, loss, metric] = build_model(inputs,
                                            build_model_fun,
                                            labels,
                                            collections,
                                            summary,
                                            check_output_shape,
                                            check_input_shape,
                                            reuse,
                                            scope,
                                            **model_config)
    if isinstance(filename, str):
        if layers.core.__graph__ is not None and \
          layers.core.__graph__ is not False:
            layers.core.export_graph(filename)
        else:
            print('WARNING: cannot save graph to file {}.'
                  'To do that, run {} first'.format(filename,
                          colors.red('sigma.engine.set_print(True)')))
    #----- train configuration -----#
    optimizer = ops.optimizers.get(optimizer, **optimizer_config)

    def _experiment(args):
        train_config = helpers.arg2dict(args,
                                        ['epochs', 'save_mode',
                                         'save_target', 'filename'])
        test_config = helpers.arg2dict(args,
                                       ['predop', 'batch_size', 'axis',
                                        'nclass', 'savedir', 'reuse',
                                        'name', 'scope'])
        # the process of training networks
        if args.verbose:
            helpers.print_args(args)

        #----- beg re-configurations -----#
        checkpoint = args.checkpoint
        log = args.log
        expid = args.eid
        if args.auto_timestamp:
            timestamp = helpers.timestamp(fmt='%Y%m%d%H%M%S', split=None)
            if expid is None:
                expid = timestamp
            else:
                expid = '{}-{}'.format(expid, timestamp)
        if expid is not None:
            if checkpoint is None:
                checkpoint = '{}/ckpt/main'.format(expid)
            else:
                checkpoint = '{}/{}/ckpt/main'.format(checkpoint, expid)
            if log is None:
                log = '{}/log'.format(expid)
            else:
                log = '{}/{}/log'.format(log, expid)
        else:
            if checkpoint is not None:
                checkpoint = '{}/ckpt/main'.format(checkpoint)
            if log is not None:
                log = '{}/log'.format(log)
        #----- end re-configurations -----#

        #----- get rid of some parameters in dictionary -----#
        train_config_keys = train_config.keys()
        for key in ['generator', 'iterations',
                    'optimizer', 'loss', 'metric',
                    'valid_gen', 'valid_iters', 'config']:
            if key in train_config_keys:
                print('`{}` in parser not allowed. will be removed'
                      .format(colors.red(key)))
                del train_config[key]
        #----- check parameters not allowed for sigma.train
        for key in list(train_config_keys):
            if key not in ['epochs', 'save_mode', 'save_target','filename']:
                print('sigma.train contains no parameter `{}`. will be removed'
                      .format(colors.red(key)))
                del train_config[key]
        #********** check filename existance if not None **********
        if expid is not None and args.filename is None:
            train_config['filename'] = expid + '.rec'
            if args.checkpoint is not None:
                train_config['filename'] = os.path.join(args.checkpoint, train_config['filename'])
        if train_config['filename'] is not None:
            if os.path.isfile(train_config['filename']):
                go = input('Filename to record loss / metric exists.'
                           ' Overwrite? <Y/N> ')
                if go != 'Y':
                    print('OK, exit program.')
                    exit(0)
                else:
                    print('OK, good luck.')
            elif os.path.isdir(train_config['filename']):
                raise ValueError('`{}` is a directory not file'
                                 .format(colors.red(train_config['filename'])))
            else:
                dirname = os.path.dirname(train_config['filename'])
                if len(dirname) != 0 and not os.path.exists(dirname):
                    mkdir = input('Parent directory {} not exists. Create? <Y/N>'
                                  .format(colors.red(dirname)))
                    if mkdir != 'Y':
                        print('OK, exit program.')
                        exit(0)
                    else:
                        os.makedirs(dirname)
                        print('OK, created. Good luck')
        #*************************************************************

        loss_op = ops.losses.get(loss)
        optimization_op = optimizer.minimize(loss_op)
        train_op = {'optimizer':optimization_op, 'loss':loss_op}
        valid_op = {'loss':loss_op}

        sess, saver, summarize, writer = session(config=config,
                                                 checkpoint=checkpoint,
                                                 log=log,
                                                 debug=args.debug,
                                                 address=args.address)
        checkpoint_save_op = checkpoint_save(checkpoint,
                                             saver,
                                             write_meta_graph=False,
                                             verbose=False)
        train(sess,
              generator,
              iterations,
              train_op,
              valid_op,
              metric,
              valid_gen,
              valid_iters,
              writer,
              summarize,
              checkpoint_save_op,
              **train_config)

        if test_gen is not None and test_iters is not None:
            predict(sess, inference, generator=test_gen, iterations=test_iters, checkpoint=checkpoint, **test_config)

        ops.core.close_summary_writer(writer)
        ops.core.close_session(sess)

    return _experiment
