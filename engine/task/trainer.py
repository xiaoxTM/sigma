import torch
from sigma.engine import misc
from sigma.engine.task.runner import Runner

import typing


# import logging
#
# logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
# logger = logging.getLogger(__file__)
# logger.setLevel(logging.INFO)


def train_distributed(rank: int, world_size: int, args: typing.Doct[str, typing.Any]):
    torch.cuda.set_device(rank)
    import torch.distributed as dist
    env = args['train'].get('distributed', {})
    misc.distributed_setup(rank=rank,
                           world_size=world_size,
                           backend=env['backend'],
                           init_method=args['init_method'])
    Runner(args).train()
    dist.destroy_process_group()


def train(args: typing.Doct[str, typing.Any]):
    Runner(args).train()


def valid(args: typing.Doct[str, typing.Any]):
    runner = Runner(args)

    runner.evaluate(split='valid')


def drun(args: typing.Doct[str, typing.Any]):
    """ dry-run dataset, i.e., test loading dataset
    """
    runner = Runner(args)

    runner.dry_run()
