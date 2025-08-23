import typing
import argparse
import os.path
import os
import logging

from sigma.core.utils import config as cfg
from sigma.core.utils import utils
from sigma.core.fontstyles import colors

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def prepare(task: str, kwargs: typing.Dict[str, typing.Any]):
    """
    Prepares the task and kwargs for further processing.
    Args:
        task (str): The task name.
        kwargs (dict): Additional keyword arguments.
    Returns:
        None
    Raises:
        None
    """

    path = kwargs.get('workspace', None)
    if path is None:
        timestamp = utils.timestamp()
        if kwargs['exp_name'] is not None:
            timestamp = '{}-{}'.format(kwargs['exp_name'], timestamp)
        path = os.path.join(kwargs['root'], task, timestamp)
        kwargs.update({'workspace': path})
    os.makedirs(path, exist_ok=True)
    logger.info('setting workspace to {}'.format(colors.blue(path)))
    cfg.save(os.path.join(kwargs['workspace'], 'config.yaml'), kwargs)
    model_name = kwargs['model']['name']
    if os.path.exists('models/architectures/{}'.format(model_name)):
        from sigma.engine.misc import ios
        ios.archive(os.path.join(kwargs['workspace'], '{}.zip'.format(model_name)),
                    'models/architectures/{}'.format(model_name),
                    pattern='__pycache__')
    else:
        logger.info('no models/architectures/{} fold found, skip archive files'.format(model_name))


def train(kwargs: typing.Doct[str, typing.Any]):
    """
    Trains a model using the specified arguments.
    Args:
        kwargs (dict): A dictionary containing the arguments for training.
    Returns:
        None
    """
    prepare('train', kwargs)

    # fix the seed if given
    if kwargs['seed'] is not None:
        utils.set_seed(kwargs['seed'])
        import torch
        torch.manual_seed(kwargs['seed'])
        import torch.cuda
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kwargs['seed'])
        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from sigma.engine.task import trainer

    import torch.autograd
    torch.autograd.set_detect_anomaly(kwargs['detect_anomaly'])

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and kwargs['run_mode'] == 'distributed':
        ''' specified and satisfied distributed data parallel mode
        '''
        world_size = kwargs['train'].get('distributed', {}).get('world_size', None)
        if world_size is None:
            world_size = num_gpus
        import torch.multiprocessing as mp
        mp.spawn(trainer.train_distributed,
                 args=(world_size, kwargs),
                 nprocs=world_size,
                 join=True)
    else:
        if num_gpus == 1:
            ''' single gpu mode
            '''
            kwargs['run_mode'] = 'single'
        # either single gpu or data-parallel model, run this branch
        trainer.train(kwargs)


def drun(kwargs: typing.Doct[str, typing.Any]):
    prepare('dryrun', kwargs)

    from sigma.engine.task import trainer

    import torch.autograd
    torch.autograd.set_detect_anomaly(kwargs['detect_anomaly'])

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and kwargs['run_mode'] == 'distributed':
        ''' specified and satisfied distributed data parallel mode
        '''
        world_size = kwargs['train'].get('distributed', {}).get('world_size', None)
        if world_size is None:
            world_size = num_gpus
        import torch.multiprocessing as mp
        mp.spawn(trainer.train_distributed,
                 args=(world_size, kwargs),
                 nprocs=world_size,
                 join=True)
    else:
        if num_gpus == 1:
            ''' single gpu mode
            '''
            kwargs['run_mode'] = 'single'
        # either single gpu or data-parallel model, run this branch

    from sigma.engine.task import trainer
    trainer.drun(kwargs)


def valid(kwargs: typing.Doct[str, typing.Any]):
    from sigma.engine import misc
    misc.process_intrinsics(kwargs)

    checkpath = kwargs['train']['checkpoint']['path']
    assert checkpath is not None and os.path.exists(checkpath), 'checkpoint not given or not exist:{}'.format(checkpath)
    prepare('valid', kwargs)
    kwargs['run_mode'] = 'single'

    from sigma.engine.task import trainer
    trainer.valid(kwargs)


def infer(kwargs: typing.Doct[str, typing.Any]):
    from sigma.engine import misc
    misc.process_intrinsics(kwargs)

    if kwargs['path'] is not None or kwargs['infer']['path'] is not None:
        cp = kwargs['infer']['checkpoint']
        assert cp is not None and os.path.exists(cp), 'checkpoint not given or not exist: {}'.format(cp)
    prepare('infer', kwargs)
    kwargs['run_mode'] = 'single'

    from sigma.engine.task import inferencer
    inferencer.infer(kwargs)


def export(kwargs: typing.Doct[str, typing.Any]):
    prepare('export', kwargs)

    from sigma.engine import task
    task.export(kwargs)


def live(kwargs: typing.Doct[str, typing.Any]):
    from sigma.engine import misc
    from sigma.engine import task
    misc.process_intrinsics(kwargs)
    task.live(kwargs)


def build_argparses() -> typing.Dict[str, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()

    # config: optional
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default=None,
                        help='config files for constructing model and run')

    # devices: optional
    parser.add_argument('-d',
                        '--devices',
                        type=str,
                        default=None,
                        help='GPU devices to run')

    parser.add_argument('-n',
                        '--exp-name',
                        type=str,
                        default=None,
                        help='the name of a run (experiment)')

    parser.add_argument('-D',
                        '--deterministic',
                        action='store_true',
                        default=False,
                        help='switch to deterministic for cudnn')

    parser.add_argument('-m',
                        '--mixed_precision',
                        action='store_true',
                        default=False,
                        help='enable mix precision training or not')

    parser.add_argument('-a',
                        '--detect-anomaly',
                        action='store_true',
                        default=False,
                        help='set to detect anomaly')

    parser.add_argument('-w',
                        '--workspace',
                        type=str,
                        default=None,
                        help='workspace where files are stored')

    # add subparsers
    subparsers = parser.add_subparsers()

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--run-mode',
                              type=str,
                              default='single',
                              choices=['default', 'single', 'data-parallel', 'distributed'],
                              help='the run mode. data-parallel for DataParallel mode, \
                                    distributed for DistributedDataParallel mode')
    train_parser.add_argument('--seed', type=int, default=191008, help='seed for randomization to reproduce the results')
    train_parser.add_argument('--init-method',
                              type=str,
                              default='env://')

    # dry run
    drun_parser = subparsers.add_parser('drun')
    drun_parser.set_defaults(func=drun)
    drun_parser.add_argument('--run-mode',
                             type=str,
                             default='single',
                             choices=['default', 'single', 'data-parallel', 'distributed'],
                             help='the run mode. data-parallel for DataParallel mode, \
                                    distributed for DistributedDataParallel mode')
    drun_parser.add_argument('--init-method',
                             type=str,
                             default='env://')

    # valid
    valid_parser = subparsers.add_parser('valid')
    valid_parser.set_defaults(func=valid)

    # infer
    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument('--path',
                              type=str,
                              default=None,
                              help='path of the files to infer')
    infer_parser.add_argument('--augments',
                              type=str,
                              nargs='+',
                              default=[],
                              help='augments to apply to the dataset')
    infer_parser.add_argument('--labels',
                              type=str,
                              nargs='+',
                              default=[],
                              help='labels to analyze given segment maps')
    infer_parser.set_defaults(func=infer)

    # live
    live_parser = subparsers.add_parser('live')
    live_parser.add_argument('--source',
                             type=int,
                             default=0,
                             help='camera source')
    live_parser.set_defaults(func=live)

    # export
    export_parser = subparsers.add_parser('export')
    export_parser.add_argument('--onnx-file',
                               type=str,
                               default=None,
                               help='path where onnx file to be stored')
    export_parser.add_argument('--inputs',
                               type=str,
                               default=None,
                               help='inputs for onnx. None for random generation. split by `:` for multiple inputs')
    export_parser.add_argument('--input-names',
                               type=str,
                               nargs='+',
                               default=['left', 'right'],
                               help='the input nodes of the onnx')
    export_parser.add_argument('--output-names',
                               type=str,
                               default=['output'],
                               nargs='+',
                               help='the output nodes of the onnx')
    export_parser.add_argument('--opset-version',
                               type=int,
                               default=17,
                               help='opset version of the onnx')
    export_parser.set_defaults(func=export)

    return {
        'parser': parser,
        'train_parser': train_parser,
        'drun_parser': drun_parser,
        'valid_parser': valid_parser,
        'infer_parser': infer_parser,
        'live_parser': live_parser,
        'export_parser': export_parser
    }


def main(parsers: typing.Dict[str, argparse.ArgumentParser] = None):
    if parsers is None:
        parsers = build_argparses()
    args, remains = parsers['parser'].parse_known_args()

    # NOTE:
    # make sure no torch relative packages are imported
    # before assign CUDA_VISIBLE_DEVICES variable
    if args.devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        logger.info('configuring CUDA_VISIBLE_DEVICES to {}'.format(colors.green(args.devices)))
    del args.devices

    logger.info('configuration file: {}'.format(colors.blue(args.config)))

    import yaml
    yaml.add_constructor('!inc', cfg.AdvancedYaml('configs'), yaml.SafeLoader)
    config = cfg.load(args.config, Loader=yaml.SafeLoader)
    # merge args with config
    config = cfg.merge_args(config, args)

    # delete the default functions of the parser
    del config['func']

    from sigma.engine import misc
    for remain in remains:
        # if the value is a number or bool type, should be decorated by '#'
        # e.g., 'train.epochs=#10'
        #       'train.checkpoint.path=/path/to/checkpoint'
        remain = misc.parse_remain_arg(remain)
        exec(f'config{remain}')

    os.environ['DREAMVISION_HARDWARE_PLATFORM'] = config['model']['params']['hardware_platform']
    import torch
    logger.info('PyTorch version: <{}>'.format(colors.green(torch.__version__)))
    args.func(config)


if __name__ == '__main__':
    main()
