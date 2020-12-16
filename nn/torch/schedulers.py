import torch
import torch.optim.lr_scheduler as lr
from sigma import parse_params, version
from sigma.fontstyles import colors

__schedulers__ = {'labmda':lr.LambdaLR,
                  'step': lr.StepLR,
                  'multistep': lr.MultiStepLR,
                  'ms': lr.MultiStepLR,
                  'exponential': lr.ExponentialLR,
                  'exp': lr.ExponentialLR,
                  'cosineannealing': lr.CosineAnnealingLR,
                  'ca': lr.CosineAnnealingLR,
                  'reduceonplateau': lr.ReduceLROnPlateau,
                  'rop': lr.ReduceLROnPlateau,
                  'cyclic': lr.CyclicLR}

__batch__ = [lr.CyclicLR]
__epoch__ = [lr.LambdaLR, lr.StepLR, lr.MultiStepLR, lr.ExponentialLR, lr.CosineAnnealingLR, lr.ReduceLROnPlateau]

if version.ge(torch.__version__, '1.4.0'):
    __schedulers__.update({'multiplicative': lr.MultiplicativeLR,
                           'mp': lr.MultiplicativeLR})
    __epoch__.extend([lr.MultiplicativeLR])
if version.gt(torch.__version__, '1.3.0'):
    __schedulers__.update({'onecycle': lr.OneCycleLR,
                           'oc': lr.OneCycleLR,
                           'cosineannealingwarmrestart': lr.CosineAnnealingWarmRestarts,
                           'cawr': lr.CosineAnnealingWarmRestarts})
    __batch__.extend([lr.OneCycleLR, lr.CosineAnnealingWarmRestarts])

def get(scheduler, optimizer):
    if scheduler is None or isinstance(scheduler, lr._LRScheduler):
        return scheduler
    elif isinstance(scheduler, str):
        scheduler_type, params = parse_params(scheduler)
        scheduler_type = scheduler_type.lower()
        assert scheduler_type in __schedulers__.keys(), 'scheduler type {} not support'.format(scheduler_type)
        assert scheduler_type not in ['reduceonplateau'], 'currently {} not support'.format(scheduler_type)
        return __schedulers__[scheduler_type](optimizer, **params)
    else:
        raise TypeError('cannot convert type {} into Scheduler'.format(colors.red(type(scheduler))))

def register(key, scheduler):
    assert key is not None and scheduler is not None, 'both key and scheduler can not be none'
    global __schedulers__
    assert key not in __schedulers__.keys(), 'key {} already registered'.format(key)
    assert isinstance(scheduler, lr._LRScheduler), 'scheduler must be an instance of _LRScheduler, given {}'.format(scheduler)
    __schedulers__.update({key:scheduler})
