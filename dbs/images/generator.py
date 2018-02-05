from __future__ import print_function
import numpy as np
import os.path
import logging
from ..utils import load_filename_from_dir
from .ios import load_from_list
from sigma.helpers.nput import one_hot

######################################################
def segment_data_augmentator(clip=[0, 255],
                             fliplr=0.25,
                             flipud=0.25,
                             saturation=0.5,
                             contrast=0.5,
                             brightness=0.5,
                             lighting_std=0.5):
    """ data augmentator for segmentation task
        saturation / contrast / brightness / lighting_std: either None (without this operation),
        or vector (size of 2, [probability of doing this operation, variance parameter]) or scalar

        Attributes
        ----------
        clip : list / tuple or int
               used to clip value when augmentating with
               `saturation`, `contrast`, `brightness` or `lighting_std`
        fliplr : float
               probability for horizontal flipping augmentation operation
        flipup : float
               probability for vertical flipping augmentation operation
        saturation : float
               probability for saturation augmentation operation
        contrast : float
               probability for contrast augmentation operation
        brightness : float
               probability for brightness augmentation operation
    """
    if isinstance(saturation, (list, tuple, np.ndarray)):
        assert len(saturation) == 2, '`saturation` as list/tuple '\
                                     'must have size of two. given {}'\
                                     .format(len(saturation))
    elif isinstance(saturation, (float, np.float, np.float32, np.float64)):
        saturation = [saturation, saturation]
    elif saturation is not None:
        raise TypeError('`saturation` support only scalar or list. given {}'
                        .format(type(saturation)))
    if isinstance(contrast, (list, tuple, np.ndarray)):
        assert len(contrast) == 2, '`contrast` as list/tuple '\
                                   'must have size of two. given {}'\
                                   .format(len(contrast))
    elif isinstance(contrast, (float, np.float, np.float32, np.float64)):
        contrast = [contrast, contrast]
    elif contrast is not None:
        raise TypeError('`contrast` support only scalar or list. given {}'
                        .format(type(contrast)))
    if isinstance(brightness, (list, tuple, np.ndarray)):
        assert len(brightness) == 2, '`brightness` as list/tuple '\
                                     'must have size of two. given {}'\
                                     .format(len(brightness))
    elif isinstance(brightness, (float, np.float, np.float32, np.float64)):
        brightness = [brightness, brightness]
    elif brightness is not None:
        raise TypeError('`brightness` support only scalar '
                        'or list. given {}'.format(type(brightness)))
    color_jitter = []
    if saturation is not None:
        color_jitter.append(_saturation)
    if contrast is not None:
        color_jitter.append(_contrast)
    if brightness is not None:
        color_jitter.append(_brightness)

    if isinstance(lighting_std, (list, tuple, np.ndarray)):
        assert len(lighting_std) == 2, '`lighting_std` as list/tuple '\
                                       'must have size of two. given {}'\
                                       .format(len(lighting_std))
    elif isinstance(lighting_std, (float, np.float, np.float32, np.float64)):
        lighting_std = [lighting_std, lighting_std]
    elif lighting_std is not None:
        raise TypeError('`lighting_std` support only scalar or list. '
                        'given {}'.format(type(lighting_std)))

    if isinstance(clip, (list, tuple)):
        assert len(clip) == 2, '`clip` of list/tuple must '\
                               'have length of 2. given {}'.format(len(clip))
    elif isinstance(clip, int):
        clip = [0, clip]
    else:
        raise TypeError('`clip` can only be list/tuple or int. '
                        'given {}'.format(type(clip)))

    def _grayscale(rgb):
        return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    def _saturation(rgb):
        ret = rgb
        if np.random.random() <= saturation[0]:
            logging.debug('augmentation/saturation')
            gs = _grayscale(rgb)
            alpha = 2 * np.random.random() * saturation[1]
            alpha += 1 - saturation[1]
            rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
            ret = np.clip(rgb, clip[0], clip[1])
        return ret
    def _brightness(rgb):
        ret = rgb
        if np.random.random() <= brightness[0]:
            logging.debug('augmentation/brightness')
            alpha = 2 * np.random.random() * brightness[1]
            alpha += 1 - saturation[1]
            rgb = rgb * alpha
            ret = np.clip(rgb, clip[0], clip[1])
        return ret
    def _contrast(rgb):
        ret = rgb
        if np.random.random() <= contrast[0]:
            logging.debug('augmentation/contrast')
            gs = _grayscale(rgb).mean() * np.ones_like(rgb)
            alpha = 2 * np.random.random() * contrast[1]
            alpha += 1 - contrast[1]
            rgb = rgb * alpha + (1 - alpha) * gs
            ret = np.clip(rgb, clip[0], clip[1])
        return ret
    def _lighting(image):
        ret = image
        if lighting_std is not None and np.random.random() <= lighting_std[0]:
            logging.debug('augmentation/lighting')
            cov = np.cov(image.reshape(-1, 3) / 255.0, rowvar=False)
            eigval, eigvec = np.linalg.eigh(cov)
            noise = np.random.randn(3) * lighting_std[1]
            noise = eigvec.dot(eigval * noise) * 255
            image += noise
            ret = np.clip(image, clip[0], clip[1])
        return ret
    def _flip_horizontal(image, truth, append):
        logging.debug('augmentation/flip-horizontal')
        image_flipped, truth_flipped = np.fliplr(image), np.fliplr(truth)
        if append:
            image_flipped = np.append(image_flipped, image, axis=0)
            truth_flipped = np.append(truth_flipped, truth, axis=0)
        return image_flipped, truth_flipped

    def _flip_vertical(image, truth, append):
        logging.debug('augmentation/flip-vertical')
        image_flipped, truth_flipped = np.flipud(image), np.flipud(truth)
        if append:
            image_flipped = np.append(image_flipped, image, axis=0)
            truth_flipped = np.append(truth_flipped, truth, axis=0)
        return image_flipped, truth_flipped

    def _augment(samples, labels, append):
        if len(color_jitter) > 0:
            random.shuffle(color_jitter)
            for jitter in color_jitter:
                samples = jitter(samples)
        samples = _lighting(samples)
        if fliplr is not None and fliplr >= np.random.random():
            samples, labels = _flip_horizontal(samples, labels, append)
        if flipud is not None and flipud >= np.random.random():
            samples, labels = _flip_vertical(samples, labels, append)
        return samples, labels
    return _augment


#####################################################
# functions / classes for segmentation task
#####################################################
#####################################################
""" generator for image data with augmentation
"""
def segment_data_generator(self, nclass, filelist, batch_size,
                           xtensor=None,
                           ytensor=None,
                           basepath=None,
                           size=(320, 320),
                           void_label=0,
                           one_hot=True,
                           scale=1.0,
                           center=True,
                           strides=0.5,
                           mode='crop',
                           fliplr=None,
                           flipud=None,
                           saturation=None,
                           contrast=None,
                           brightness=None,
                           lighting_std=None,
                           clip=[0, 255],
                           spi=None,
                           num=None,
                           sep=' ',
                           preprocess_input=None):
    """ initialization of generator

        Attributes
        ----------
        nclass : int
                     number of class to generate
        filelist : list / tuple / str / dict
                   if list / tuple, dataset is given in the order of
                       [train, valid, test]
                   if str, dataset is given by train samples
                   if dict, dataset is given in form of key: value
                   Example:
                       'vallid': /path/to/valid/samples.txt
        batch_size : int
                     number of samples for each batch
        basepath : str or None
                   if str, the samples to load will be
                       os.path.join(basepath, samplefilename)
                   if None, samples to load is samplefilename
        size : tuple or None
               if tuple, images will be resized or cropped to `size`
                   according to the value of `mode`
               if None, loaded images will have the original size
                   **NOTE**: when given None, images to be load
                   must have the same size
        void_label : int
                     label of class treated as background (void)
        one_hot : bool
                  return one hot format of ground truth if True
        scale : float
                scale factor for each images
                Example:
                    scale = 1 / 255.0 will scale images from [0, 255] to [0, 1]
        center : bool
                 if True, images will crop according to the center of images
                     **NOTE**: this is useful ONLY when `mode` is crop
        strides : float or list / tuple
                  steps to move when crop
        mode : str (case sensitive)
               mode to resize images. must be value of
               ['crop', 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic']
        fliplr / flipud / saturation / contrast / brightness / lighting_std / clip
               are parameters passed to SegmentDataAugmentator
               see @SegmentDataAugmentator for details
        batch_size_variable : bool
               generator will generate batches with different size if True,
               otherwise generate same batch size as specified by `batch_size`
               ** NOTE ** this is used ONLY in `crop` mode
            preprocess_input : callable function for preprocess input before yielding data
    """
    train_samples, train_labels = None, None
    valid_samples, valid_labels = None, None
    test_samples , test_labels  = None, None
    eval_samples , eval_labels  = None, None
    if xtensor is None:
        xtensor = 'xtensor'
    if ytensor is None:
        ytensor = 'ytensor'
    if isinstance(filelist, (list, tuple)):
        if not len(filelist) in [1, 2, 3, 4]:
            raise ValueError('file list must have 1, 2, 3 or 4 '
                             'length if given as list / tuple. given {}'
                             .format(len(filelist)))
        train_samples, train_labels = load_filename(filelist[0],
                                                    num=num, sep=sep)
        train_samples = np.asarray(train_samples, dtype=np.string_)
        if train_labels is None:
            raise ValueError('ground truth is not given to train phase')
        train_labels = np.asarray(train_labels, dtype=np.string_)
        if len(filelist) >= 2:
            valid_samples, valid_labels = load_filename(filelist[1],
                                                        num=num, sep=sep)
            valid_samples = np.asarray(valid_samples, dtype=np.string_)
            if valid_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            valid_labels = np.asarray(valid_labels, dtype=np.string_)
        if len(filelist) >= 3:
            test_samples, test_labels = load_filename(filelist[2],
                                                      num=num, sep=sep)
            test_samples = np.asarray(test_samples, dtype=np.string_)
            if test_labels is not None:
                test_labels = np.asarray(test_labels, dtype=np.string_)
        if len(filelist) == 4:
            eval_samples, eval_labels = load_filename(filelist[3],
                                                      num=num, sep=sep)
            eval_samples = np.asarray(eval_samples, dtype=np.string_)
            if eval_labels is not None:
                eval_labels = np.asarray(eval_labels, dtype=np.string_)
    elif isinstance(filelist, str):
        train_samples, train_labels = load_filename(filelist,
                                                    num=num, sep=sep)
        train_samples = np.asarray(train_samples, dtype=np.string_)
        if train_labels is None:
            raise ValueError('ground truth is not given to train phase')
        train_labels = np.asarray(train_labels, dtype=np.string_)
    elif isinstance(filelist, dict):
        for key in filelist.keys():
            if key not in ['train', 'valid', 'test', 'eval']:
                raise ValueError('Unsupported phase: {}'.format(key))
        if 'train' in filelist.keys():
            train_samples, train_labels = load_filename(filelist['train'],
                                                        num=num, sep=sep)
            train_samples = np.asarray(train_samples, dtype=np.string_)
            if train_labels is None:
                raise ValueError('ground truth is not given to train phase')
            train_labels = np.asarray(train_labels, dtype=np.string_)
        if 'valid' in filelist.keys():
            valid_samples, valid_labels = load_filename(filelist['valid'],
                                                        num=num, sep=sep)
            valid_samples = np.asarray(valid_samples, dtype=np.string_)
            if valid_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            valid_labels = np.asarray(valid_labels, dtype=np.string_)
        if 'test' in filelist.keys():
            test_samples, test_labels = load_filename(filelist['test'],
                                                      num=num, sep=sep)
            test_samples = np.asarray(test_samples, dtype=np.string_)
            if test_labels is not None:
                test_labels = np.asarray(test_labels, dtype=np.string_)
        if 'eval' in filelist.keys():
            eval_samples, eval_labels = load_filename(filelist['eval'],
                                                      num=num, sep=sep)
            eval_samples = np.asarray(eval_samples, dtype=np.string_)
            if eval_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            eval_labels = np.asarray(eval_labels, dtype=np.string_)
    else:
        raise TypeError('file list can only be list/tuple '
                        '(for providing train/valid dataset list file)'
                        ' or string (for providing train dataset list file)'
                        '. given {}'.format(type(filelist)))
    if size is None:
        size = size
    elif isinstance(size, (list, tuple)):
        assert len(size) == 2, 'size of list/tuple must have length'\
                               ' of 2. given {}'.format(len(size))
        size = size
    elif isinstance(size, int):
        size = [size, size]
    else:
        raise TypeError('size can only be list/tuple or int. '
                        'given {}'.format(type(size)))
    if mode not in ('crop', 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'):
        raise ValueError('Bad mode: {}'.format(mode))
    if void_label >= nclass:
        raise ValueError('background / void label[{}] must '
                         'less than number of class[{}]'
                         .format(void_label, nclass))
    asarray = True
    _augmentator = segment_data_augmentator(clip,
                                            fliplr,
                                            flipud,
                                            saturation,
                                            contrast,
                                            brightness,
                                            lighting_std)
    if preprocess_input is None:
        preprocess_input = lambda x: x
    def _nsamples(x):
        if x is None:
            return 0
        return len(x)
    def _generate(phase='train'):
        """generate dataset batches

            This function is a python generator, to call, use `next()` function

            Examples
            ----------
            samples, labels = next(generate('train'))
            samples = next(generate('test'))

            Attributes
            ----------
            phase : str
                    specifying phase of 'train', 'valid', 'test' or 'eval'
                    for phase of 'train', 'valid' or 'eval', ground truth must be provided
                    for 'test' phase, ground truth is not mandatory, and will be ignored if provided

            Returns
            ----------
                numpy ndarray of batches of samples.
                for phase of 'train', 'valid' or 'eval', return '(samples, labels)'
                for phase 'test', return 'labels'
        """
        assert phase in ['train', 'valid', 'test', 'eval'],\
               'phase can only be `train`, `valid`, `test` and `eval`. '\
               'given `{}`'.format(phase)
        while True: # for debug
            if phase == 'train':
                index = np.arange(len(train_samples))
                np.random.shuffle(index)
                samples, labels = train_samples[index], train_labels[index]
            elif phase == 'valid':
                assert valid_samples is not None, \
                       'valid_samples is not given for valid phase'
                assert valid_labels is not None, \
                       'valid_samples is not given for valid phase'
                index = np.arange(len(valid_samples))
                np.random.shuffle(index)
                samples, labels = valid_samples[index], valid_labels[index]
            elif phase == 'test':
                assert test_samples is not None, \
                       'test_samples is not given for test phase'
                samples, labels = test_samples, test_labels
                if labels is None:
                    labels = [None] * len(test_samples)
            else:
                assert eval_samples is not None, \
                       'eval_samples is not given for eval phase'
                samples, labels = eval_samples, eval_labels
            inputs = []
            targets = []
            nsample = 0
            for sample, truth in zip(samples, labels):
                label_path = truth
                if basepath is not None:
                    image_path = os.path.join(basepath, sample.decode('UTF-8'))
                    if truth is not None:
                        label_path = os.path.join(basepath, truth.decode('UTF-8'))
                else:
                    image_path = sample.decode('UTF-8')
                    if truth is not None:
                        label_path = truth.decode('UTF-8')
                logging.debug('loading image')
                image, label = load_image(image_path,
                                          label_path,
                                          size,
                                          asarray,
                                          scale,
                                          center,
                                          strides,
                                          mode,
                                          spi,
                                          void_label)
                logging.debug('done')
                if len(image) < 0:
                    continue
                if phase in ['train', 'valid', 'eval'] and label is None:
                    raise ValueError('ground truth must be given for phase '
                                     '`{}`'.format(phase))

                if label is not None and _one_hot:
                    label = dense_one_hot(label, nclass)
                # augmentation for train ONLY
                if phase == 'train':
                    image, label = _augment(image, label, spi!=1)

                nsample += 1
                inputs.extend(image)
                if label is not None:
                    targets.extend(label)

                if nsample == batch_size:
                    inputs = np.asarray(inputs)
                    if channel_axis == 1:
                        inputs = np.transpose(inputs, (0, 3, 1, 2))
                    if len(targets) != 0:
                        targets = np.asarray(targets)
                        if channel_axis == 1 and one_hot:
                            targets = np.transpose(targets, (0, 3, 1, 2))
                        logging.debug('sample shape: {}, label shape: {}'
                                      .format(inputs.shape, targets.shape))
                    else:
                        logging.debug('sample shapes: {}'.format(_inputs.shape))
                    inputs = []
                    targets = []
                    nsample = 0
                    inputs = _preprocess_input(inputs)
                    if phase == 'test': # test phase
                        yield {xtensor:inputs}
                    else: # train / valid phase
                        yield {xtensor:inputs, ytensor:targets}
    return _generate, \
           map(_nsamples,
               [train_samples, valid_samples, test_samples, eval_samples])


def generator(imagepath, batch_size,
              xtensor=None,
              ytensor=None,
              gtpath=None,
              gtext=None,
              size=None,
              asarray=True,
              scale=1.0,
              center=True,
              strides=0.5,
              mode='crop',
              spi=None,
              void_label=0,
              multiprocess=1,
              color_mode='RGB',
              basepath=None,
              sep=' ',
              num=None,
              namefilter=None,
              nclass=None,
              onehot=False):
    """ generator to generate batch of images
        Attributes
        ----------
            imagepath : string
                       source image directory or file that list image names
            batch_size : int
                         size to generate images for once
            imagepath : string
                    ground truth image directory
                    (used if imagepath is directory)
            gtext : string
                    ground truth image filename suffix
                    (used if imagepath is directory)
            size : None / list / tuple
                   size to resize after loading image
                   None will not resize
            asarray : bool
                      return list or numpy.array
            scale : float
                    scale image values when load
            center : bool
                     center image when cropping
                     ONLY for mode = 'crop'
            strides : int / float
                      strides when crop images
            spi : int
                  samples per image
            void_label : int
                         background label for class label
            multiprocess : int
                           number of processes when loading images
            color_mode : string
                         color mode when load image
                         e.g., 'RGB', 'RGBA'
            basepath : string
                       basepath for loading filenames
                       (usef if imagepath is filename)
            sep : string
                  separator for image filename and groundtruth filename
                  (usef if imagepath is filename)
            num : None:
                  how many images to load.
                  None for loading all images
            namefilter : None / callable
                         used to filter images by name
            nclass : None / int
                     number of class when onehot ground truth
            onehot : bool
                     whether should onehot ground truth or not
    """
    if os.path.isdir(imagepath):
        filelist, gtlist = load_filename_from_dir(imagepath, gtpath, gtext,
                                                  num, namefilter)
    elif os.path.isfile(imagepath):
        filelist, gtlist = load_filename(imagepath, num, basepath, sep,
                                         namefilter)
    filelist = np.asarray(filelist, dtype=np.string_)
    if gtlist is not None:
        gtlist = np.asarray(gtlist, dtype=np.string_)
    length = len(filelist)
    index = np.arange(length, dtype=np.int32)
    iterations = max(int(length / batch_size), 1)
    if xtensor is None:
        xtensor = 'xtensor'
    if ytensor is None:
        ytensor = 'ytensor'
    def _generate():
        while True:
            np.random.shuffle(index)
            # get the previous `batch-size` samples
            for iteration in range(iterations):
                beg = iteration * batch_size
                end = min(beg + batch_size, length)
                x_list = filelist[index[beg:end]]
                y_list = None
                if gtlist is not None:
                    y_list = gtlist[index[beg:end]]
                x, y = load_from_list(x_list, y_list, size,
                                      asarray, scale, center,
                                      strides, mode, spi,
                                      void_label, multiprocess, color_mode)
                if onehot:
                    if y is not None:
                        y = one_hot(y, nclass)
                yield {xtensor:x, ytensor:y}, iteration
    return _generate(), length, iterations


def next_batch(dataset, iteration, batch_size, shuffle=True):
    if isinstance(dataset, (tuple, list)):
        if len(dataset) == 1:
            x = dataset[0]
            y = None
        elif len(dataset) == 2:
            x, y = dataset
        else:
            raise ValueError('`dataset` must have length '
                             'of 2 when given list/tuple')
    elif isinstance(dataset, np.ndarray):
        x = dataset
        y = None
    else:
        raise TypeError('`dataset` must be list/tuple or np.ndarray. '
                         'given {}'.format(type(dataset)))
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if iteration == 0 and shuffle:
        idx = np.arange(len(x), dtype=np.int32)
        np.random.shuffle(idx)
        x = x[idx]
        if y is not None:
            y = y[idx]
    beg = batch_size * iteration
    if beg >= len(x):
        beg = 0
    end = min(beg + batch_size, len(x))
    if y is None:
        return x[beg:end]
    return x[beg:end], y[beg:end]


def make_generator(x,
                   y=None,
                   xtensor=None,
                   ytensor=None,
                   batch_size=32,
                   shuffle=True,
                   nclass=None):
    """ make generator from dataset
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if y is not None:
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError('`x` and `y` must have same length. '
                             'given {} vs {}'.format(colors.red(len(x)),
                                                     colors.green(len(y))))
        if nclass is not None:
            y = one_hot(y, nclass)
    length = len(x)
    idx = np.arange(length, dtype=np.int32)
    iterations = max(int(length / batch_size), 1)
    if xtensor is None:
        xtensor = 'xtensor'
    if ytensor is None:
        ytensor = 'ytensor'
    def _generate():
        while True:
            if shuffle:
                np.random.shuffle(idx)
            for iteration in range(iterations):
                beg = iteration * batch_size
                if beg > length:
                    beg = length - batch_size
                end = max(min(beg + batch_size, length), beg)
                if y is not None:
                    yield {xtensor:x[idx[beg:end]], ytensor:y[idx[beg:end]]}, iteration
                else:
                    yield {xtensor:x[idx[beg:end]]}, iteration
    return _generate(), length, iterations
