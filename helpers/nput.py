import numpy as np
import multiprocessing as mp
import pickle
import gzip
import os.path
import logging
from scipy import misc as sm

""" numpy utils
"""


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
    """crop single image
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
    """crop list of images
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


#####################################################
def load_image(filename,
               gtname=None,
               size=None,
               asarray=True,
               scale=1.0,
               center=True,
               strides=0.5,
               mode='crop',
               spi=None,
               void_label=0,
               color_mode='RGB'):
    """
    filename: source image filename
    gtname: ground truth image filename if not none. if not need, set to none
    mode: one of {'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'} for resize
          and 'crop' for crop image
    center: only available for 'crop' mode
    strides: only available for 'crop' mode
    """
    assert scale > 0
    assert os.path.isfile(filename), 'file {} not found'.format(filename)
    logging.debug('loading {}'.format(filename))
    sample = sm.imread(filename, mode=color_mode)
    assert len(sample.shape) == 2 or len(sample.shape) == 3
    sample = sample.astype(np.float32) * scale

    assert mode in ('crop', 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic')

    label = None
    if gtname is not None:
        assert os.path.isfile(gtname), 'file {} not found'.format(gtname)
        label  = sm.imread(gtname, mode=color_mode)
        assert len(label.shape) == 2

    if size is not None:
        assert isinstance(size, (list, tuple))
        assert len(size) == 2 or len(size) == 3

        if mode == 'crop':
            sample = crop_image(sample, size, center, strides)
            if gtname is not None:
                label = crop_image(label, size, center, strides)
        else:
            if len(sample.shape) == len(size):
                if sample.shape != size:
                    sample = [sm.imresize(sample, size, interp=mode)]
                    if gtname is not None:
                        label  = [sm.imresize(label , size, interp=mode)]
            else:
                if len(size) == 2:
                    sample = [sm.imresize(sample,(size[0], size[1],
                                                  sample.shape[2]),
                                                  interp=mode)
                                                  .astype(np.float32)]
                    if gtname is not None:
                        label  = [sm.imresize(label , (size[0], size[1],
                                                       sample[0].shape[2]),
                                                       interp=mode)]
                else:
                    sample = [sm.imresize(sample, (size[0], size[1]),
                                                   interp=mode)
                                                   .astype(np.float32)]
                    if gtname is not None:
                        label  = [sm.imresize(label , (size[0], size[1]), interp=mode)]
    else:
        sample = [sample]
        if gtname is not None:
            label = [label]

    if label is not None:
        # remove samples contain only background
        for idx, (s, l) in enumerate(zip(sample, label)):
            if np.sum(l != void_label) == 0:
                del sample[idx]
                del label[idx]
                logging.warning('warning: remove image contains background ONLY')

    if len(sample) != 0 and asarray:
        sample = np.asarray(sample)
        if label is not None:
            label = np.asarray(label)

    nsample = len(sample)
    if spi is not None and spi < nsample:
        index = np.arange(nsample, dtype='int')
        np.random.shuffle(index)
        if spi == 1:
            sample = [sample[index[0]]]
        else:
            sample = sample[index[:spi]]

        if label is not None:
            if spi == 1:
                label = [label[index[0]]]
            else:
                label = label[index[:spi]]

    return sample, label


def load_image_multiprocess_worker(parameters):
    filename, gtname, size, asarray, scale, center, strides, \
              mode, spi, void_label, color_mode = parameters
    return load_image(filename, gtname, size, asarray, scale,\
                      center, strides, mode, spi, void_label, color_mode)


def load_from_list_multiprocess(filelist, gtlist, multiprocess,
                                size=None,
                                asarray=True,
                                scale=1.0,
                                center=True,
                                strides=0.5,
                                mode='crop',
                                spi=None,
                                void_label=0,
                                color_mode='RGB'):
    assert isinstance(filelist, (list, tuple))
    assert gtlist is None or isinstance(gtlist, (list, tuple))

    samples = []
    labels  = []

    if gtlist is None:
        gtlist = [None] * len(filelist)
    assert len(filelist) == len(gtlist)
    counter = 0
    if multiprocess < 1:
        multiprocess = mp.cpu_count()
    parameters = [None] * multiprocess
    worker = mp.Pool(processes=multiprocess)
    for f, g in zip(filelist, gtlist):
        if g is not None and not os.path.isfile(g):
            continue
        parameters[counter] = [f, g, size, asarray, scale, center,
                               strides, mode, spi, void_label, color_mode]
        counter += 1
        if counter == multiprocess:
            samples_and_labels = worker.map(load_image_multiprocess_worker, parameters)
            counter = 0
            for sample, label in samples_and_labels:
                if len(sample) > 0:
                    samples.extend(sample)
                    if label is not None:
                        labels.extend(label)
    if counter != 0:
        samples_and_labels = worker.map(load_image_multiprocess_worker, parameters[:counter])
        for sample, label in samples_and_labels:
            if len(sample) > 0:
                samples.extend(sample)
                if label is not None:
                    labels.extend(label)
    worker.close()
    if asarray:
        samples = np.asarray(samples)
        if gtlist[0] is not None:
            labels = np.asarray(labels)
    return samples, labels


def load_from_list_simple(filelist, gtlist,
                          size=None,
                          asarray=True,
                          scale=1.0,
                          center=True,
                          strides=0.5,
                          mode='crop',
                          spi=None,
                          void_label=0,
                          color_mode='RGB'):
    """load image in single process
    Attributes
    ----------
    spi: None or int
         samples per image. only used in `crop` mode sampling
    """
    assert isinstance(filelist, (list, tuple, np.ndarray))
    assert gtlist is None or isinstance(gtlist, (list, tuple))
    samples = []
    labels  = []
    if gtlist is None:
        for f in filelist:
            sample, _ = load_image(f, None, size, asarray, scale, center,
                                   strides, mode, spi, void_label, color_mode)
            if len(sample) > 0:
                samples.extend(sample)
    else:
        assert len(filelist) == len(gtlist)
        for f, g in zip(filelist, gtlist):
            if not os.path.isfile(g):
                logging.warning('ground truth {} not exist, file {} ignored'.format(g, f))
                continue
            sample, label = load_image(f, g, size, asarray, scale, center,
                                       strides, mode, spi, void_label, color_mode)
            if len(sample) > 0:
                samples.extend(sample)
                labels.extend(label)
    if asarray:
        samples = np.asarray(samples)
        if gtlist is not None:
            labels = np.asarray(labels)
    return samples, labels


##################################################################################
def load_from_list(filelist, gtlist,
                   size=None,
                   asarray=True,
                   scale=1.0,
                   center=True,
                   strides=0.5,
                   mode='crop',
                   spi=None,
                   void_label=0,
                   multiprocess=1,
                   color_mode='RGB'):
    """
    Descriptions
    ============
        load samples and ground truth labels from list `filelist`

    Attributes
    ----------
        filelist : list of string
                   filenames to load
        gtlist : list of string
                 ground truth labels of the corresponding filelist to load.
                 if None, none ground truth labels will be load
        size : list / tuple
               size of images to load
               if None, will load the original image without resize and crop
        asarray : boolean
                  return numpy.ndarray if True, else list of images
        scale : float
                scale loaded images (samples). For example, scale = 1 / 255.0 normalize image to [0, 1]
        center : boolean
                 center crop or not if mode = 'crop'
        strides : float or list
                  strides for cropping if mode = 'crop'
        mode : string
               mode to resize image to size. see @load_image for more details
        spi : int
              samples per one image if mode == 'crop'
        void_label : int
                     ignore class label
        multiprocess : int
                       processes to load image
        color_mode : string
                     color spaces used to load image (see @https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html)
                     - L    : 8 bit pixels, black and white
                     - P    : 8-bit pixels, mapped to any other mode using a color palette
                     - RGB  : 3x8-bit pixels, true color
                     - RGBA : 4x6-bit pixels, true color with transparency mask
                     - CMYK : 4x8-bit pixels, color separation
                     - YCbCr: 3x8-bit pixels, color video format
                     - I    : 32-bit signed integer pixels
                     - F    : 32-bit floating point pixels
    Returns:
    ---------
        list of images [and ground truth if gtlist is not None] if asarray is False
        numpy.ndarray of images [and ground truth if gtlist is not None] if asarray is True

    """
    if multiprocess != 1:
        return load_from_list_multiprocess(filelist, gtlist, multiprocess,
                                           size, asarray, scale, center,
                                           strides, mode, spi, void_label,
                                           color_mode)
    else:
        return load_from_list_simple(filelist, gtlist, size, asarray, scale,
                                     center, strides, mode, spi, void_label,
                                     color_mode)


##############################################################################
def load_filename(listname,
                  num=None,
                  basepath=None,
                  sep=' ',
                  namefilter=None):
    """
    load filename from listname

    Attributes
    ----------
        listname : string
                   file name where to load filename
        num : int
              number of filename to load. None means all filename
        sep : string
              separator of filename and the corresponding ground truth filename
        namefilter : function
              filters to filter out filename not need
              namefilter need filename and, return True if accept, or False if want to filter out
    """
    fn = open(listname, 'r')
    lines = fn.readlines()
    fn.close()
    filelist = []
    gtlist = []

    if namefilter is None:
        namefilter = lambda x: True

    if num is not None:
       lines = lines[:num]
    if basepath is not None:
        if isinstance(basepath, str):
            filepath = basepath
            gtpath = basepath
        elif isinstance(basepath, (list, tuple)):
            if len(basepath) != 2:
                raise ValueError('base path must have length of 2. given {}'
                                 .format(len(basepath)))
            filepath = basepath[0]
            gtpath = basepath[1]
        else:
            raise TypeError('basepath can only be "None",'
                            '"list / tuple" and "str". given {}'
                            .format(type(basepath)))
    for line in lines:
        files = line.rstrip('\n\r')
        if sep in files:
            files = files.split(sep)
        if isinstance(files, str):
            if basepath is not None:
                files = os.path.join(filepath, files)
            if namefilter(files):
                filelist.append(files)
        elif isinstance(files, list):
            if len(files) == 2:
                if basepath is not None:
                    files[0] = os.path.join(filepath, files[0])
                    files[1] = os.path.join(gtpath, files[1])
                if namefilter(files[0]):
                    filelist.append(files[0])
                    gtlist.append(files[1])
            else:
                raise ValueError('files should have length of 2 for'
                                 ' [sample, ground truth]. given length {}'
                                 .format(len(files)))
        else:
            raise TypeError('files should be str or list. given {}'
                            .format(type(files)))
    if len(gtlist) == 0:
        gtlist = None
    return filelist, gtlist


##############################################################################
def load_filename_from_dir(imagedir,
                           gtdir=None,
                           gtext=None,
                           num=None,
                           namefilter=None):
    assert os.path.isdir(imagedir)
    assert gtdir is None or (os.path.isdir(gtdir) and gtext is not None)

    filelist = []
    gtlist   = []
    num_load = 0

    if namefilter is None:
        namefilter = lambda x: True

    if num is None:
        for root, dirs, files in os.walk(imagedir):
            for f in files:
                if namefilter(os.path.join(root, f)):
                    filelist.append(os.path.join(root, f))
                    if gtdir is not None:
                        gtlist.append(os.path.join(gtdir, f.rsplit('.', 1)[0]+'.'+gtext))
    else:
        if not isinstance(num, 'int'):
            raise TypeError('`num` must be int. given {}'.format(type(num)))
        if num <= 0:
            raise ValueError('`num` must be greater than zero. given {}'
                             .format(num))
        for root, dirs, files in os.walk(imagedir):
            for f in files:
                if num_load >= num:
                    break
                if namefilter(os.path.join(root, f)):
                    num_load += 1
                    filelist.append(os.path.join(root, f))
                    if gtdir is not None:
                        gtlist.append(os.path.join(gtdir, f.rsplit('.', 1)[0]+'.'+gtext))

    if len(gtlist) == 0:
        gtlist = None

    return filelist, gtlist


##############################################################################
def load_from_file(listname,
                   size=None,
                   asarray=True,
                   scale=1.0,
                   center=True,
                   strides=0.5,
                   mode='crop',
                   basepath=None,
                   num=None,
                   spi=None,
                   sep=' ',
                   void_label=0,
                   multiprocess=1,
                   namefilter=None,
                   color_mode='RGB'):
    filelist, gtlist = load_filename(listname, num, basepath, sep, namefilter)
    return load_from_list(filelist, gtlist, size, asarray, scale, center,
                          strides, mode, spi, void_label, multiprocess, color_mode=color_mode)


################################################################################
def load_from_dir(imagedir,
                  gtdir=None,
                  gtext=None,
                  size=None,
                  asarray=True,
                  scale=1.0,
                  center=True,
                  strides=0.5,
                  mode='crop',
                  num=None,
                  spi=None,
                  void_label=0,
                  multiprocess=1,
                  namefilter=None,
                  color_mode='RGB'):
    """
    size can be [width of image, height of image] for grayscale
    or [num of channel, width of image, height of image]
    patterns are pattern sqeuences compiled by re.compile function
    all the pattern in patterns will be applied recursively to extract
    the ID (label) of the corresponding input image
    e.g.,
       string = pattern[0].search(string).group(0)
       string = pattern[1].search(string).group(0)
       string = pattern[2].search(string).group(0)
                ...
    """
    filelist, gtlist = load_filename_from_dir(imagedir, gtdir, gtext, num, namefilter)

    return load_from_list(filelist, gtlist, size, asarray, scale, center,
                          strides, mode, spi, void_label,
                          multiprocess=1, color_mode=color_mode)


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
    for a in array:
        assert isinstance(a, np.ndarray)
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
