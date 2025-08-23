from torch.optim.lr_scheduler import *

from sigma.core.utils.params import parse_params

__schedulers__ = {
    'lambda': LambdaLR,
    'multiplicative': MultiplicativeLR,
    'mp': MultiplicativeLR,
    'multistep': MultiStepLR,
    'ms': MultiStepLR,
    'constant': ConstantLR,
    'const': ConstantLR,
    'linear': LinearLR,
    'exponential': ExponentialLR,
    'exp': ExponentialLR,
    'polynomial': PolynomialLR,
    'poly': PolynomialLR,
    'cosineannealing': CosineAnnealingLR,
    'ca': CosineAnnealingLR,
    'chainedscheduler': ChainedScheduler,
    'cs': ChainedScheduler,
    'sequential': SequentialLR,
    'seq': SequentialLR,
    'reduceonplateau': ReduceLROnPlateau,
    'rp': ReduceLROnPlateau,
    'cyclic': CyclicLR,
    'one': OneCycleLR,
    'oc': OneCycleLR,
    'cosineannealingwarmrestarts': CosineAnnealingWarmRestarts,
    'cawr': CosineAnnealingWarmRestarts
}


def get(name: str, optimizer: optim.Optimizer) -> LRScheduler:
    assert optimizer is not None
    key, params = parse_params(name)
    assert key in __schedulers__.keys(), f'scheduler `{key}` not implemented yet'
    return __schedulers__[key](optimizer, **params)


def register(name: str, scheduler: LRScheduler):
    assert name.lower() not in __schedulers__, f'scheduler `{name.lower()}` already registered'
    __schedulers__[name.lower()] = scheduler
