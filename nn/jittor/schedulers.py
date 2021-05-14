import jittor
from jittor import optim
from jittor import lr_scheduler as lrs
from sigma import parse_params, version
from sigma.fontstyles import colors

__schedulers__ = {'labmda':optim.LambdaLR,
                  #'step': lrs.StepLR,
                  'multistep': lrs.MultiStepLR,
                  'ms': lrs.MultiStepLR,
                  #'exponential': lrs.ExponentialLR,
                  #'exp': lrs.ExponentialLR,
                  'cosineannealing': lrs.CosineAnnealingLR,
                  'ca': lrs.CosineAnnealingLR,
                  'reduceonplateau': lrs.ReduceLROnPlateau,
                  'rop': lrs.ReduceLROnPlateau,
                  #'cyclic': lrs.CyclicLR}
                  }

#__batch__ = [lrs.CyclicLR]
#__epoch__ = [lrs.LambdaLR, lrs.StepLR, lrs.MultiStepLR, lrs.ExponentialLR, lrs.CosineAnnealingLR, lrs.ReduceLROnPlateau]

#if version.ge(torch.__version__, '1.4.0'):
#    __schedulers__.update({'multiplicative': lrs.MultiplicativeLR,
#                           'mp': lrs.MultiplicativeLR})
#    __epoch__.extend([lrs.MultiplicativeLR])
#if version.gt(torch.__version__, '1.3.0'):
#    __schedulers__.update({'onecycle': lrs.OneCycleLR,
#                           'oc': lrs.OneCycleLR,
#                           'cosineannealingwarmrestart': lrs.CosineAnnealingWarmRestarts,
#                           'cawr': lrs.CosineAnnealingWarmRestarts})
#    __batch__.extend([lrs.OneCycleLR, lrs.CosineAnnealingWarmRestarts])
#
def get(scheduler, optimizer):
    if scheduler is None or isinstance(scheduler, optim.LRScheduler):
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
    assert isinstance(scheduler, lrs._LRScheduler), 'scheduler must be an instance of _LRScheduler, given {}'.format(scheduler)
    __schedulers__.update({key:scheduler})
