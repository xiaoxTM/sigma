from .augments import *

from sigma.core.utils import parse_params

__augments__ = {
}


def get(names: typing.List[str]):
    # for am in ams:
    #     assert  am['name'].lower() in __augments__.keys()
    # aug = [__augments__[name'].lower()](**am['params']) for am in ams]
    augs = []
    if names is not None:
        for name in names:
            key, params = parse_params(name)
            assert key.lower() in __augments__.keys(), 'augmentation {} not implemented'.format(key)
            augs.append(__augments__[key](**params))
    return Compose(augs)


def register(name: str, func: typing.Callable):
    assert name.lower() not in __augments__.keys(), 'augmentation {} already registered'.format(name.lower())
    __augments__[name.lower()] = func
