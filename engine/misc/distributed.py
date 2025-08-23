import torch
import torch.distributed as dist
from sigma.engine import misc

import os
import logging

from sigma.core.fontstyles import colors


logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def sync_mean(tensor: torch.Tensor, distributed: bool, **kwargs) -> torch.Tensor:
    if distributed:
        dist.all_reduce(tensor, **kwargs)
        tensor /= dist.get_world_size()
    return tensor


def distributed_setup(rank: int, world_size: int, backend: str, init_method: str) -> None:
    """
    init_method: env://ip:port | file:///path/to/shared/file | tcp://ip:port
    """
    method = init_method.split('//')
    assert len(method) == 2, 'wrong init method format: {}'.format(colors.red(init_method))
    if method[0] == 'env:':
        addr_info = method[1].split(':')
        if len(addr_info) == 0:
            port = '19108'
            address = 'localhost'
        elif len(addr_info) == 1:
            port = '19108'
            address = addr_info[0]
            if address == '':
                address = 'localhost'
        elif len(addr_info) == 2:
            address, port = addr_info
            if port == '':
                port = '19108'
        else:
            raise ValueError('unsupported method pattern: {}'.format(init_method))
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port
        # init_method = method
        misc.log(logger.info,
                 'initialize distributed mode with method: {}, address: {}:{}'.format(colors.blue(init_method),
                                                                                      colors.blue(address),
                                                                                      colors.blue(port)))

    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)
