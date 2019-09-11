from . import simple
from . import novel
from . import multibranch
from . import multibranch_v1
from . import multibranch_v2
from . import ae_rec
from . import pointnet
from . import pcn
from . import pcn_multigpu
from . import caps_gnn

def build_net(arch, inputs, labels, loss='margin_loss', *args, **kwargs):
    return eval('{}.build_net(inputs, labels, loss, *args, **kwargs)'.format(arch))
