from . import simple
from . import novel
from . import multibranch

def build_net(arch, inputs, *args, **kwargs):
    return eval('{}.build_net(inputs, *args, **kwargs)'.format(arch))
