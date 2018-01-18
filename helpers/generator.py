from __future__ import print_function
import numpy as np
import os.path
import logging
from .nput import load_filename_from_dir, load_from_list, one_hot

#####################################################
# functions / classes for segmentation task
#####################################################
#####################################################
"""generator for image data with augmentation
"""
def segment_data_generator(self, nclass, filelist, batch_size,
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
    if isinstance(filelist, (list, tuple)):
        if not len(filelist) in [1, 2, 3, 4]:
            raise ValueError('file list must have 1, 2, 3 or 4 length if given as list / tuple. given {}'.format(len(filelist)))
        _train_samples, _train_labels = load_filename(filelist[0], num=num, sep=sep)
        _train_samples = np.asarray(_train_samples, dtype=np.string_)
        if _train_labels is None:
            raise ValueError('ground truth is not given to train phase')
        _train_labels = np.asarray(_train_labels, dtype=np.string_)
        if len(filelist) >= 2:
            _valid_samples, _valid_labels = load_filename(filelist[1], num=num, sep=sep)
            _valid_samples = np.asarray(_valid_samples, dtype=np.string_)
            if _valid_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            _valid_labels = np.asarray(_valid_labels, dtype=np.string_)
        if len(filelist) >= 3:
            _test_samples, _test_labels = load_filename(filelist[2], num=num, sep=sep)
            _test_samples = np.asarray(_test_samples, dtype=np.string_)
            if _test_labels is not None:
                _test_labels = np.asarray(_test_labels, dtype=np.string_)
        if len(filelist) == 4:
            _eval_samples, _eval_labels = load_filename(filelist[3], num=num, sep=sep)
            _eval_samples = np.asarray(_eval_samples, dtype=np.string_)
            if _eval_labels is not None:
                _eval_labels = np.asarray(_eval_labels, dtype=np.string_)
    elif isinstance(filelist, str):
        _train_samples, _train_labels = load_filename(filelist, num=num, sep=sep)
        _train_samples = np.asarray(_train_samples, dtype=np.string_)
        if _train_labels is None:
            raise ValueError('ground truth is not given to train phase')
        _train_labels = np.asarray(_train_labels, dtype=np.string_)
    elif isinstance(filelist, dict):
        for key in filelist.keys():
            if key not in ['train', 'valid', 'test', 'eval']:
                raise ValueError('Unsupported phase: {}'.format(key))
        if 'train' in filelist.keys():
            _train_samples, _train_labels = load_filename(filelist['train'], num=num, sep=sep)
            _train_samples = np.asarray(_train_samples, dtype=np.string_)
            if _train_labels is None:
                raise ValueError('ground truth is not given to train phase')
            _train_labels = np.asarray(_train_labels, dtype=np.string_)
        if 'valid' in filelist.keys():
            _valid_samples, _valid_labels = load_filename(filelist['valid'], num=num, sep=sep)
            _valid_samples = np.asarray(_valid_samples, dtype=np.string_)
            if _valid_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            _valid_labels = np.asarray(_valid_labels, dtype=np.string_)
        if 'test' in filelist.keys():
            _test_samples, _test_labels = load_filename(filelist['test'], num=num, sep=sep)
            _test_samples = np.asarray(_test_samples, dtype=np.string_)
            if _test_labels is not None:
                _test_labels = np.asarray(_test_labels, dtype=np.string_)
        if 'eval' in filelist.keys():
            _eval_samples, _eval_labels = load_filename(filelist['eval'], num=num, sep=sep)
            _eval_samples = np.asarray(_eval_samples, dtype=np.string_)
            if _eval_labels is None:
                raise ValueError('ground truth is not given to valid phase')
            _eval_labels = np.asarray(_eval_labels, dtype=np.string_)
    else:
        raise TypeError('file list can only be list/tuple (for providing train/valid dataset list file) or string (for providing train dataset list file). given {}'.format(type(filelist)))

    _batch_size = batch_size
    _basepath = basepath
    _nclass = nclass
    if size is None:
        _size = size
    elif isinstance(size, (list, tuple)):
        assert len(size) == 2, 'size of list/tuple must have length of 2. given {}'.format(len(size))
        _size = size
    elif isinstance(size, int):
        _size = [size, size]
    else:
        raise TypeError('size can only be list/tuple or int. given {}'.format(type(size)))

    if mode not in ('crop', 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'):
        raise ValueError('Bad mode: {}'.format(mode))
    _mode = mode

    if void_label >= nclass:
        raise ValueError('background / void label[{}] must less than number of class[{}]'.format(void_label, nclass))
    _void_label = void_label

    _scale = scale
    _center = center
    _strides = strides
    _spi = spi
    _one_hot = one_hot
    _asarray = True
    #_augmentator = SegmentDataAugmentator(clip, fliplr, flipud, saturation, contrast, brightness, lighting_std)

    _preprocess_input = preprocess_input if preprocess_input is not None else lambda x: x

    @property
    def train_nsamples(self):
        if hasattr(self, 'train_samples'):
            return len(_train_samples)
        return 0

    @property
    def valid_nsamples(self):
        if hasattr(self, 'valid_samples'):
            return len(_valid_samples)
        return 0

    @property
    def test_nsamples(self):
        if hasattr(self, 'test_samples'):
            return len(_test_samples)
        return 0

    @property
    def eval_nsamples(self):
        if hasattr(self, 'eval_samples'):
            return len(_eval_samples)
        return 0

    @property
    def background(self):
        return _void_label

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
        assert phase in ['train', 'valid', 'test', 'eval'], 'phase can only be `train`, `valid`, `test` and `eval`. given `{}`'.format(phase)
        while True: # for debug
            if phase == 'train':
                index = np.arange(len(_train_samples))
                np.random.shuffle(index)
                samples, labels = _train_samples[index], _train_labels[index]
            elif phase == 'valid':
                assert _valid_samples is not None, 'valid_samples is not given for valid phase'
                assert _valid_labels is not None, 'valid_samples is not given for valid phase'
                index = np.arange(len(valid_samples))
                np.random.shuffle(index)
                samples, labels = _valid_samples[index], _valid_labels[index]
            elif phase == 'test':
                assert test_samples is not None, 'test_samples is not given for test phase'
                samples, labels = _test_samples, _test_labels
                if labels is None:
                    labels = [None] * len(test_samples)
            else:
                assert eval_samples is not None, 'eval_samples is not given for eval phase'
                samples, labels = _eval_samples, _eval_labels

            inputs = []
            targets = []
            nsample = 0
            for sample, truth in zip(samples, labels):
                label_path = truth
                if _basepath is not None:
                    image_path = os.path.join(_basepath, sample.decode('UTF-8'))
                    if truth is not None:
                        label_path = os.path.join(_basepath, truth.decode('UTF-8'))
                else:
                    image_path = sample.decode('UTF-8')
                    if truth is not None:
                        label_path = truth.decode('UTF-8')
                logging.debug('loading image')
                image, label = load_image(image_path, label_path, _size, _asarray, _scale, _center, _strides, _mode, _spi, _void_label)
                logging.debug('done')
                if len(image) < 0:
                    continue
                if phase in ['train', 'valid', 'eval'] and label is None:
                    raise ValueError('ground truth must be given for phase `{}`'.format(phase))

                if label is not None and _one_hot:
                    label = dense_one_hot(label, _nclass)
                # augmentation for train ONLY
                if phase == 'train':
                    image, label = _augmentator.augment(image, label, _spi!=1)

                nsample += 1
                inputs.extend(image)
                if label is not None:
                    targets.extend(label)

                if nsample == _batch_size:
                    _inputs = np.asarray(inputs)
                    if _channel_axis == 1:
                        _inputs = np.transpose(_inputs, (0, 3, 1, 2))
                    if len(targets) != 0:
                        _targets = np.asarray(targets)
                        if _channel_axis == 1 and _one_hot:
                            _targets = np.transpose(_targets, (0, 3, 1, 2))
                        logging.debug('sample shape: {}, label shape: {}'.format(_inputs.shape, _targets.shape))
                    else:
                        logging.debug('sample shapes: {}'.format(_inputs.shape))
                    inputs = []
                    targets = []
                    nsample = 0
                    _inputs = _preprocess_input(_inputs)
                    if phase == 'test': # test phase
                        yield _inputs
                    else: # train / valid phase
                        yield _inputs, _targets


######################################################
class SegmentDataAugmentator():
    """data augmentator for segmentation task
    """
    def __init__(self, clip=[0, 255], fliplr=0.25, flipud=0.25, saturation=0.5,
                 contrast=0.5, brightness=0.5, lighting_std=0.5):
        """
                saturation / contrast / brightness / lighting_std: either None (without this operation),
                or vector (size of 2, [probability of doing this operation, variance parameter]) or scalar
            Attributes
            ----------
            clip : list / tuple or int
                   used to clip value when augmentating with `saturation`, `contrast`, `brightness` or `lighting_std`
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
        self.fliplr = fliplr
        self.flipud = flipud
        if saturation is None:
            self.saturation = saturation
        elif isinstance(saturation, (list, tuple, np.ndarray)):
            assert len(saturation) == 2, '`saturation` as list/tuple must have size of two. given {}'.format(len(saturation))
            self.saturation = saturation
        elif isinstance(saturation, (float, np.float, np.float32, np.float64)):
            self.saturation = [saturation, saturation]
        else:
            raise TypeError('`saturation` support only scalar or list. given {}'.format(type(saturation)))

        if contrast is None:
            self.contrast = contrast
        elif isinstance(contrast, (list, tuple, np.ndarray)):
            assert len(contrast) == 2, '`contrast` as list/tuple must have size of two. given {}'.format(len(contrast))
            self.contrast = contrast
        elif isinstance(contrast, (float, np.float, np.float32, np.float64)):
            self.contrast = [contrast, contrast]
        else:
            raise TypeError('`contrast` support only scalar or list. given {}'.format(type(contrast)))

        if brightness is None:
            self.brightness = brightness
        elif isinstance(brightness, (list, tuple, np.ndarray)):
            assert len(brightness) == 2, '`brightness` as list/tuple must have size of two. given {}'.format(len(brightness))
            self.brightness = brightness
        elif isinstance(brightness, (float, np.float, np.float32, np.float64)):
            self.brightness = [brightness, brightness]
        else:
            raise TypeError('`brightness` support only scalar or list. given {}'.format(type(brightness)))

        self.color_jitter = []
        if self.saturation is not None:
            self.color_jitter.append(self._saturation)
        if contrast is not None:
            self.color_jitter.append(self._contrast)
        if brightness is not None:
            self.color_jitter.append(self._brightness)

        if lighting_std is None:
            self.lighting_std = lighting_std
        elif isinstance(lighting_std, (list, tuple, np.ndarray)):
            assert len(lighting_std) == 2, '`lighting_std` as list/tuple must have size of two. given {}'.format(len(lighting_std))
            self.lighting_std = lighting_std
        elif isinstance(lighting_std, (float, np.float, np.float32, np.float64)):
            self.lighting_std = [lighting_std, lighting_std]
        else:
            raise TypeError('`lighting_std` support only scalar or list. given {}'.format(type(lighting_std)))

        if isinstance(clip, (list, tuple)):
            assert len(clip) == 2, '`clip` of list/tuple must have length of 2. given {}'.format(len(clip))
            self.clip = clip
        elif isinstance(clip, int):
            self.clip = [0, clip]
        else:
            raise TypeError('`clip` can only be list/tuple or int. given {}'.format(type(clip)))

    def _grayscale(self, rgb):
        return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

    def _saturation(self, rgb):
        ret = rgb
        if np.random.random() <= self.saturation[0]:
            logging.debug('augmentation/saturation')
            gs = self._grayscale(rgb)
            alpha = 2 * np.random.random() * self.saturation[1]
            alpha += 1 - self.saturation[1]
            rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
            ret = np.clip(rgb, self.clip[0], self.clip[1])
        return ret

    def _brightness(self, rgb):
        ret = rgb
        if np.random.random() <= self.brightness[0]:
            logging.debug('augmentation/brightness')
            alpha = 2 * np.random.random() * self.brightness[1]
            alpha += 1 - self.saturation[1]
            rgb = rgb * alpha
            ret = np.clip(rgb, self.clip[0], self.clip[1])
        return ret

    def _contrast(self, rgb):
        ret = rgb
        if np.random.random() <= self.contrast[0]:
            logging.debug('augmentation/contrast')
            gs = self._grayscale(rgb).mean() * np.ones_like(rgb)
            alpha = 2 * np.random.random() * self.contrast[1]
            alpha += 1 - self.contrast[1]
            rgb = rgb * alpha + (1 - alpha) * gs
            ret = np.clip(rgb, self.clip[0], self.clip[1])
        return ret

    def _lighting(self, image):
        ret = image
        if self.lighting_std is not None and np.random.random() <= self.lighting_std[0]:
            logging.debug('augmentation/lighting')
            cov = np.cov(image.reshape(-1, 3) / 255.0, rowvar=False)
            eigval, eigvec = np.linalg.eigh(cov)
            noise = np.random.randn(3) * self.lighting_std[1]
            noise = eigvec.dot(eigval * noise) * 255
            image += noise
            ret = np.clip(image, self.clip[0], self.clip[1])
        return ret

    def _flip_horizontal(self, image, truth, append):
        logging.debug('augmentation/flip-horizontal')
        image_flipped, truth_flipped = np.fliplr(image), np.fliplr(truth)
        if append:
            image_flipped = np.append(image_flipped, image, axis=0)
            truth_flipped = np.append(truth_flipped, truth, axis=0)
        return image_flipped, truth_flipped

    def _flip_vertical(self, image, truth, append):
        logging.debug('augmentation/flip-vertical')
        image_flipped, truth_flipped = np.flipud(image), np.flipud(truth)
        if append:
            image_flipped = np.append(image_flipped, image, axis=0)
            truth_flipped = np.append(truth_flipped, truth, axis=0)
        return image_flipped, truth_flipped

    def augment(self, samples, labels, append):
        if len(self.color_jitter) > 0:
            random.shuffle(self.color_jitter)
            for jitter in self.color_jitter:
                samples = jitter(samples)
        samples = self._lighting(samples)
        if self.fliplr is not None and self.fliplr >= np.random.random():
            samples, labels = self._flip_horizontal(samples, labels, append)
        if self.flipud is not None and self.flipud >= np.random.random():
            samples, labels = self._flip_vertical(samples, labels, append)
        return samples, labels

def generator(imagedir, batch_size,
              gtdir=None,
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
              num=None,
              namefilter=None,
              nclass=None,
              onehot=False):

    filelist, gtlist = load_filename_from_dir(imagedir, gtdir, gtext,
                                              num, namefilter)
    filelist = np.asarray(filelist, dtype=np.string_)
    if gtlist is not None:
        gtlist = np.asarray(gtlist, dtype=np.string_)
    length = len(filelist)
    index = np.arange(length, dtype=np.int32)
    iterations = int(length / batch_size)
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
            yield [x, y], [iteration+1, iterations]
