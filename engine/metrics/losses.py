from torch.nn import functional as func
import torch
import typing

from sigma.core.utils.params import parse_params

__losses__ = {

}

def get(loss_list: typing.Union[str, typing.List[str]]) -> typing.Dict[str, typing.Callable]:
    assert loss_list is not None
    if not isinstance(loss_list, (list, tuple)):
        loss_list = [loss_list]

    m = {}
    for loss in loss_list:
        key, params = parse_params(loss)
        assert key in __losses__.keys(), f'loss `{key}` not implement yet'
        m.update({key: __losses__[key](**params)})
    return m


def register(name: str, loss_func: typing.Callable):
    # global __losses__
    assert name.lower() not in __losses__, f'loss `{name.lower()}` already exist'
    __losses__[name.lower()] = loss_func
