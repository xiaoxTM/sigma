import numpy as np
from PIL import Image

def imread(filename, mode='r'):
    return np.array(Image.open(filename, mode))

def imsave(filename, arr, fmt=None, **kwargs):
    Image.fromarray(arr).save(filename, fmt, **kwargs)

def imresize(arr, size, interp='bilinear'):
    mode = eval('Image.{}'.format(interp.upper()))
    Image.fromarray(arr).resize(size, mode)
