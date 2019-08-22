import sys
import os.path

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../../'))
sys.path.append(root)

from .point_capsule import *
