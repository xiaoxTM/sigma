import jittor
from sigma import parse_params
from sigma.fontstyles import colors

def l1(alpha=0.0001):
    def _l1(param_list):
        reg = 0
        for params in param_list:
            reg += jittor.sum(jittor.abs(params))
        return reg*alpha
    return _l1


def l2(beta=0.00001):
    def _l2(param_list):
        reg = 0
        for params in param_list:
           reg += jittor.sqrt(jittor.pow(params,2).sum())
        return reg*beta
    return _l2


def l1l2(alpha=0.0001, beta=0.00001):
    def _l1l2(param_list):
        reg1 = 0
        for params in param_list:
            reg1 += jittor.sum(jittor.abs(params))
            reg2 += jittor.sqrt(jittor.pow(params,2).sum())
        return (alpha*reg1 + beta*reg2)

__regularizers__ = {'l1':l1,'l2':l2,'l1l2':l1l2}


def get(reg):
    if reg is None or callable(reg):
        return reg
    elif isinstance(reg,str):
        reg = reg.strip()
        if reg in ['','null','none']:
            return None
        reg_type, params = parse_params(reg)
        reg_type = reg_type.lower()
        assert reg_type in __regularizers__.keys(), '{} regularizer type not support'.format(reg_type)
        return __regularizers__[reg_type](**params)
    else:
        raise TypeError('cannot convert type {} into regularizer'.format(colors.red(type(reg))))


def register(key, reg):
    assert key is not None and reg is not None, 'both key and loss can not be None'
    global __regularizers__
    assert key not in __regularizers__.keys(), 'regularizer `{}` alread exists'.format(key)
    assert callable(reg), 'regularizer must be a function, given {}'.format(type(loss))
    __regularizers__.update({key:reg})
