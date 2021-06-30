from torch import nn
from sigma import parse_params
from sigma.fontstyles import colors

__norms__ = {'bn1d':nn.BatchNorm1d,
             'batchnorm1d':nn.BatchNorm1d,
             'bn2d':nn.BatchNorm2d,
             'batchnorm2d':nn.BatchNorm2d,
             'bn3d':nn.BatchNorm3d,
             'batchnorm3d':nn.BatchNorm3d,
             'gn':nn.GroupNorm,
             'groupnorm':nn.GroupNorm,
             'sbn':nn.SyncBatchNorm,
             'syncbatchnorm':nn.SyncBatchNorm,
             'in1d':nn.InstanceNorm1d,
             'instancenorm1d':nn.InstanceNorm1d,
             'in2d':nn.InstanceNorm2d,
             'instancenorm2d':nn.InstanceNorm2d,
             'in3d':nn.InstanceNorm3d,
             'instancenorm3d':nn.InstanceNorm3d,
             'ln':nn.LayerNorm,
             'layernorm':nn.LayerNorm,
             'lrn':nn.LocalResponseNorm,
             'localresponsenorm':nn.LocalResponseNorm}


def get(norm, channels, dimension=None, size=None):
    if norm is None or isinstance(norm, nn.Module) or callable(norm):
        return norm
    elif isinstance(norm, str):
        norm = norm.strip()
        if norm in ['','null','none']:
            return None
        norm_type, params = parse_params(norm)
        norm_type = norm_type.lower()
        if norm_type in ['bn','batchnorm','in','instancenorm']:
            assert dimension in ['1d','2d','3d'], 'dimension must be one of "1d/2d/3d". given {}'.format(dimension)
            return __norms__[norm_type+dimension](channels, **params)
        else:
            assert norm_type in __norms__.keys(), 'norm type {} not support'.format(norm_type)
            if norm_type in ['gn','groupnorm']:
                return __norms__[norm_type](num_channels=channels, **params)
            elif norm_type in ['ln','layernorm','lrn','localresponsenorm']:
                return __norms__[norm_type](size,**params)
            else:
                return __norms__[norm_type](channels, **params)
    else:
        raise TypeError('cannot conver type {} into Normalizations'.format(colors.red(type(norm))))

def register(key, norm):
    assert key is not None and norm is not None, 'both key and norm can not be none'
    global __norms__
    assert key not in __norms__.keys(), 'key {} already registered'.format(key)
    assert isinstance(norm, nn.Module) or callable(norm), 'norm must be either an instance of nn.Module or function, given {}'.format(norm)
    __norms__.update({key:norm})
