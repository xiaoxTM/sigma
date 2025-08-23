import typing
from sigma.models import SigmaModel

__architectures__ : typing.Dict[str, SigmaModel] = {
}


def get(name: str, **params) -> SigmaModel:
    assert name.lower() in __architectures__.keys(), 'model: {} not found'.format(name.lower())
    return __architectures__[name](**params)


def register(name: str, model: SigmaModel):
    assert name.lower() not in __architectures__.keys(), 'model: {} already exists'.format(name.lower())
    __architectures__[name.lower()] = model
