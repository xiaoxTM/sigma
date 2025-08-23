from torch import optim
import re
from engine.models import SigmaModel

import logging
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


__optimizers__ = {
    'sgd': optim.SGD,
    'rprop': optim.Rprop,
    'rmsprop': optim.RMSprop,
    'radam': optim.RAdam,
    'nadam': optim.NAdam,
    'lbfgs': optim.LBFGS,
    'asgd': optim.ASGD,
    'adamax': optim.Adamax,
    'sparseadam': optim.SparseAdam,
    'adamw': optim.AdamW,
    'adam': optim.Adam,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta
}


def get(name: str, model: SigmaModel, **kwargs) -> optim.Optimizer:
    assert name in __optimizers__.keys()
    parameters = kwargs.get('parameters', [])
    if parameters is None:
        parameters = []
    if 'parameters' in kwargs.keys():
        kwargs.pop('parameters')
    trainable_parameters = {}
    for pname, param in model.named_parameters():
        if param.requires_grad:
            found = False
            for parameter in parameters:
                pattern = parameter.get('pattern')
                lr = parameter.get('lr')
                if re.search(pattern, pname) is not None:
                    d = trainable_parameters.get(pattern, {})
                    dl = d.get('params', [])
                    dl.append(param)
                    d.update({'params': dl, 'lr': lr})
                    trainable_parameters.update({pattern: d})
                    found = True
                    break
            if not found:
                d = trainable_parameters.get('__default__', {})
                dl = d.get('params', [])
                dl.append(param)
                d.update({'params': dl})
                trainable_parameters.update({'__default__': d})
            # trainable_parameters.append(param)
        else:
            logger.debug('parameter[{}] freezed'.format(pname))
    trainable_parameters = list(trainable_parameters.values())
    return __optimizers__[name](trainable_parameters, **kwargs)


def register(name: str, optimizer: optim.Optimizer):
    assert name.lower() not in __optimizers__.keys(), f"Optimizer {name.lower()} already exists in the registry."
    __optimizers__[name.lower()] = optimizer
