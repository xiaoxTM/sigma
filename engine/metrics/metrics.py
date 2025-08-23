import torch
import numpy as np
import typing
from functools import partial
from torch.nn import functional as func

from sigma.core.utils.params import parse_params


__metrics__ = {
}


def get(metric_list: typing.Union[str, typing.List[str]]) -> typing.Dict[str, typing.Callable]:
    assert metric_list is not None
    if not isinstance(metric_list, (list, tuple)):
        metric_list = [metric_list]

    m = {}
    for metric in metric_list:
        key, params = parse_params(metric)
        assert key in __metrics__.keys(), f'metrics `{key}` not implement yet'
        m.update({key: __metrics__[key](**params)})
    return m


def register(name: str, metric_func: typing.Callable):
    assert name.lower() not in __metrics__, f'metric `{name.lower()}` already exist'
    __metrics__[name.lower()] = metric_func
