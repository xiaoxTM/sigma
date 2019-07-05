"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import pickle
import gzip
from PIL import Image

""" numpy utils
"""

def imread(filename, mode='r'):
    return np.array(Image.open(filename, mode))

def imsave(filename, arr, fmt=None, **kwargs):
    Image.fromarray(arr).save(filename, fmt, **kwargs)

def imresize(arr, size, interp='bilinear'):
    mode = eval('Image.{}'.format(interp.upper()))
    Image.fromarray(arr).resize(size, mode)

######################################################
def dense_argmax(x,
                 colormap=None,
                 one_hot=True,
                 axis=-1):
    """
        x to be the form of
        1. (batch-size, nsamples, nclass) for axis =-1,
           or (batch-size, nclass, nsamples) for axis=1
        2. (batch-size, rows, cols, nclass) for axis =-1
           or (batch-size, nclass, rows, cols) for axis=1
        If not given / None: no transpose is done.

        colormap must be either dict/list/tuple or none
        1. dict: should have the form of {idx:[red, green, blue], ...}
        2. list: in form of [[red, green, blue], ...]
        3. tuple: in form of ([red, green, blue], ...)
        All above form will return a single color image.
        4. None: return one-hot label
    """
    assert isinstance(x, np.ndarray), 'x must be instance of np.ndarray'
    assert len(x.shape) == 4 or len(x.shape) == 3, \
           'x must have three or four dimensions. Given {}'.format(x.shape)
    assert not (colormap is None and one_hot is False), \
           'either one-hot be true or colormap is not None'

    assert isinstance(colormap, (dict, list, tuple)) or colormap is None, \
           'colormap must be eithe dict/list/tuple or None. given {}'\
           .format(type(colormap))
    if isinstance(axis, str):
        assert axis in ['channels_first', 'channels_last'], \
               'data format {} not support by `axis`'.format(axis)
        if axis == 'channels_last':
            axis = -1
        else:
            axis =1

    if axis == 1:
        if len(x.shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            x = np.transpose(x, (0, 2, 1))

    dense_label = np.argmax(x, axis=axis) if one_hot else x.astype(np.int32)
    if len(x.shape) == 4 or not one_hot:
        if one_hot:
            b, r, c, _ = x.shape
        else:
            b, r, c = x.shape
        if colormap is None:
            ret = np.zeros_like(x, dtype=np.int32)
            bidx, ridx, cidx = np.meshgrid(range(b), range(r), range(c))
            ret[bidx, ridx, cidx, dense_label[bidx, ridx, cidx]] = 1
            #for bidx in range(b):
            #    for ridx in range(r):
            #        for cidx in range(c):
            #            ret[bidx, ridx, cidx, dense_label[bidx, ridx, cidx]] = 1
        else:
            ret = np.zeros((b, r, c, 3), dtype=np.uint8)
            for bidx in range(b):
                for ridx in range(r):
                    for cidx in range(c):
                        ret[bidx, ridx, cidx] = colormap[dense_label[bidx, ridx, cidx]]
    else:
        # s = rows * cols
        b, s, _, = x.shape
        if colormap is None:
            ret = np.zeros_like(x, dtype=np.int32)
            bidx, sidx = np.meshgrid(range(b), range(s))
            ret[bidx, sidx, dense_label[bidx, sidx]] = 1
            #for bidx in range(b):
            #    for sidx in range(s):
            #        ret[bidx, sidx, dense_label[bidx, sidx]] = 1
        else:
            ret = np.zeros((b, s, 3), dtype=np.uint8)
            for bidx in range(b):
                for sidx in range(s):
                    ret[bidx, sidx] = colormap[dense_label[bidx, sidx]]
    return ret


#####################################################
# functions / classes for loading dataset
#####################################################
######################################################
def one_hot(y, nclass=None):
    if isinstance(y, np.ndarray):
        shape = list(y.shape)
    elif isinstance(y, list):
        y = np.asarray(y).astype(np.int32)
        shape = list(y.shape)
    #print('shape:', shape)
    if not y.dtype in [np.int32, np.int64]:
        raise TypeError('given y must be  int32 or int64 dtype. given {}'
                        .format(y.dtype))
    y = y.ravel()
    if not nclass:
        nclass = int(np.max(y) + 1)
    if nclass < np.max(y) + 1:
        raise ValueError('class {} must be equal/greater than max value {} + 1'
                         .format(nclass, np.max(y)))
    n = y.shape[0]
    cat = np.zeros((n, nclass))
    cat[np.arange(n, dtype=np.int32), y] = 1
    shape = shape + [nclass]
    return np.reshape(cat, shape)


######################################################
def crop_image(image, cropsize,
               center=True,
               strides=None):
    """ crop single image
        crop single data sample and label

        Attributes
        ----------
        cropsize: output image size
        center: whether should be center aligned or not
        strides: steps for crop image
    """
    assert isinstance(cropsize, (list, tuple))
    assert isinstance(center, bool)
    if strides is None:
        strides = cropsize
    elif isinstance(strides, float):
        strides = (int(cropsize[0]*strides), int(cropsize[1]*strides))
    assert isinstance(strides, (list, tuple))
    assert strides[0] > 0 and strides[1] > 0
    #print('strides: {}'.format(strides))
    shape = image.shape
    #print('image shape: {}'.format(shape))
    if shape[0] < cropsize[0]:
        diff = cropsize[0] - shape[0]
        left = diff // 2
        right = diff - left
        if len(shape) == 2:
            image = np.pad(image, ((left, right), (0, 0)), mode='constant')
        elif len(shape) == 3:
            image = np.pad(image, ((left, right), (0, 0), (0, 0)), mode='constant')
        else:
            raise ValueError('padding support rank 2 / 3 only')
    if shape[1] < cropsize[1]:
        diff = cropsize[1] - shape[1]
        top = diff // 2
        bottom = diff - top
        if len(shape) == 2:
            image = np.pad(image, ((0, 0), (top, bottom)), mode='constant')
        elif len(shape) == 3:
            image = np.pad(image, ((0, 0), (top, bottom), (0, 0)), mode='constant')
        else:
            raise ValueError('padding support rank 2 / 3 only')
    shape = image.shape
    images = []
    row_beg = 0
    row_end = shape[0] - cropsize[0] + 1
    col_beg = 0
    col_end = shape[1] - cropsize[1] + 1
    if center:
        nsteps = ((shape[0]-cropsize[0]) // strides[0], (shape[1]-cropsize[1]) // strides[1])
        out_shape = (nsteps[0]*strides[0]+cropsize[0], nsteps[1]*strides[1]+cropsize[1])
        #print('output shape: {}'.format(out_shape))
        row_beg = (shape[0] - out_shape[0]) // 2
        row_end = row_beg + out_shape[0] - cropsize[0] + 1
        col_beg = (shape[1] - out_shape[1]) // 2
        col_end = col_beg + out_shape[1] - cropsize[1] + 1
    #print('(rbeg, rend), (cbeg, cend) = ({}, {}), ({}, {})'.format(row_beg, row_end, col_beg, col_end))
    for row in range(row_beg, row_end, strides[0]):
        #print("=====================")
        for col in range(col_beg, col_end, strides[1]):
            if len(shape) == 2:
                images.append(image[row:row+cropsize[0], col:col+cropsize[1]])
            elif len(shape) == 3:
                images.append(image[row:row+cropsize[0], col:col+cropsize[1], :])
            else:
                raise ValueError('image shape dimension out of range [1, 3]')
    return images


######################################################
def crop(images, cropsize, center, strides, spi):
    """ crop list of images
        crop data to cropsize by strides steps
        parameters: see @crop_image
    """
    assert isinstance(images, (list, tuple, np.ndarray))
    if isinstance(images, (list, tuple)):
        images = np.asarray(images)
    assert len(images.shape) >= 3 and len(images.shape) <= 4
    imagelist = []
    for idx in range(images.shape[0]):
        if len(images.shape) == 3:
            cropped = crop_image(images[idx, :, :], cropsize, center, strides)
            imagelist.extend(cropped)
        elif len(images.shape) == 4:
            cropped = crop_image(images[idx, :, :, :], cropsize, center, strides)
            imagelist.extend(cropped)
        else:
            raise ValueError('image shape dimension out of range [2, 3]')
    nsample = len(imagelist)
    if spi is not None and spi < nsample:
        index = np.arange(nsample, dtype='int')
        np.random.shuffle(index)
        if spi == 1:
            imagelist = [imagelist[index[0]]]
        else:
            imagelist = imagelist[index[:spi]]
    return imagelist


########################################################################
def shuffle(database):
    """
    shuffle the given database
    """
    (samples, labels) = database

    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    return (samples[idx, :, :, :], labels[idx])


############################################################
def split(database, ratio):
    """
     split database into train, validate, and test subset according to
     ratio, which should have the form of [train_ratio, validate_ratio, test_ratio]
     and train_ratio + validate_ratio + test_ratio <= 1.0
     --> for sum of ratio less than 1, it means that only need sub-database and then
     --> split the sub-database into three sub-subset
    """
    assert len(ratio) == 3 and sum(ratio) <= 1
    nsamples = len(database[1])
    indice = np.int32(np.array(ratio) * nsamples)
    indice[1] = indice[0] + indice[1]
    indice[2] = indice[1] + indice[2]
    return ([database[0][:indice[0], :, :, :], database[1][:indice[0]]], \
            [database[0][indice[0]:indice[1],:,:,:], database[1][indice[0]:indice[1]]], \
            [database[0][indice[1]:indice[2],:,:,:], database[1][indice[1]:indice[2]]])


###################################################
def load_split(name, size, ratio, patterns, scale):
    """
    load and split database correspondingly
    """
    return split(shuffle(load(name, patterns, size, scale)), ratio)


################################################
def pickle_database(name, db, **kwargs):
    """
    size as the same as function load
    """
    if name.endswith('gzip'):
        dummy = gzip.open(name,'wb')
    else:
        dummy = open(name, 'wb')
    pickle.dump(db, dummy, pickle.HIGHEST_PROTOCOL, **kwargs)
    dummy.close()


####################################
def load_pickle(name, **kwargs):
    """
    load from .pkl.gz
    """
    if name.endswith('gzip'):
        pkl = gzip.open(name, 'rb')
    else:
        pkl = open(name, 'rb')
    database = pickle.load(pkl, **kwargs)
    pkl.close()
    return database


def stack(array, axis=0, interval=0, value=0.0):
    """ stack images
    """
    assert axis in [0, 1], 'axis can only be 0 or 1. given {}'.format(axis)
    assert isinstance(array, (list, tuple, np.ndarray)), \
           'array must be instance of list / tuple / numpy.ndarray. given {}'\
           .format(type(array))

    shape = array[0].shape
    for i, a in enumerate(array):
        assert isinstance(a, np.ndarray), '{} element of array is {} type rather than np.ndarray'.format(i, type(a))
        assert a.shape == shape
    length = len(array)
    # if np.ndarray of array, shape = [col, length]
    # convert it to [row, col]
    if isinstance(array, np.ndarray):
        length = array.shape[-1]
        shape = array.shape[:-1]
        array = np.transpose(array, (2, 0 ,1))
    assert length > 1, 'array length must be greater than 1. given {}'\
                       .format(length)

    dims = len(shape)
    assert dims in [2, 3]

    if axis == 0:
        if dims == 2:
            #assert isinstance(value, (np.int32, np.float32, np.float64)), 'value type: {}'.format(type(value))
            stacked = np.full((length*shape[0]+(length-1)*interval, shape[1]),
                              value, dtype=array[0].dtype)
            for idx, a in enumerate(array):
                beg = idx * (shape[0] + interval)
                end = beg + shape[0]
                stacked[beg:end, :] = a
        else:
            assert isinstance(value, (list, tuple, np.ndarray))
            assert len(value) == 3
            stacked = np.full((length*shape[0]+(length-1)*interval, shape[1], shape[2]),
                              value, dtype=array[0].dtype)
            for idx, a in enumerate(array):
                beg = idx * (shape[0] + interval)
                end = beg + shape[0]
                stacked[beg:end, :, :] = a
    else:
        if dims == 2:
            #assert isinstance(value, (np.int32, np.float32, np.float64)), 'value type: {}'.format(type(value))
            stacked = np.full((shape[0], length*shape[1]+(length-1)*interval),
                              value, dtype=array[0].dtype)
            for idx, a in enumerate(array):
                beg = idx * (shape[1] + interval)
                end = beg + shape[1]
                stacked[:, beg:end] = a
        else:
            assert isinstance(value, (list, tuple, np.ndarray))
            assert len(value) == 3
            stacked = np.full((shape[0], length*shape[1]+(length-1)*interval, shape[2]),
                              value, dtype=array[0].dtype)
            for idx, a in enumerate(array):
                beg = idx * (shape[1] + interval)
                end = beg + shape[1]
                stacked[:, beg:end, :] = a

    return stacked
