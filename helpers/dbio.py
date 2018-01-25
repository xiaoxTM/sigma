import numpy as np
import multiprocessing as mp
from scipy import misc as sm
import os
import os.path
import logging
from . import nput

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
               color_mode='RGB',
               gtloader=None):
    """
        filename: source image filename
        gtname: ground truth image filename if not none.
                if not need, set to none
        mode: one of {'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'}
              for resize and 'crop' for crop image
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
        assert callable(gtloader)
        label  = gtloader(gtname)
    if size is not None:
        assert isinstance(size, (list, tuple))
        assert len(size) == 2 or len(size) == 3
        if mode == 'crop':
            sample = nput.crop_image(sample, size, center, strides)
            if isinstance(label, np.ndarray):
                label = nput.crop_image(label, size, center, strides)
        else:
            if len(sample.shape) == len(size):
                if sample.shape != size:
                    sample = [sm.imresize(sample, size, interp=mode)]
                    if isinstance(label, np.ndarray):
                        label  = [sm.imresize(label , size, interp=mode)]
            else:
                if len(size) == 2:
                    sample = [sm.imresize(sample,(size[0], size[1],
                                                  sample.shape[2]),
                                                  interp=mode)
                                                  .astype(np.float32)]
                    if isinstance(label, np.ndarray):
                        label  = [sm.imresize(label , (size[0], size[1],
                                                       sample[0].shape[2]),
                                                       interp=mode)]
                else:
                    sample = [sm.imresize(sample, (size[0], size[1]),
                                                   interp=mode)
                                                   .astype(np.float32)]
                    if isinstance(label, np.ndarray):
                        label  = [sm.imresize(label , (size[0], size[1]), interp=mode)]
    else:
        sample = [sample]
        if isinstance(label, np.ndarray):
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
    filename, gtname, size, asarray, \
    scale, center, strides, mode, spi, \
        void_label, color_mode = parameters
    return load_image(filename, gtname, size, \
                      asarray, scale, center, \
                      strides, mode, spi, \
                      void_label, color_mode)


def load_from_list_multiprocess(filelist, gtlist, multiprocess,
                                size=None,
                                asarray=True,
                                scale=1.0,
                                center=True,
                                strides=0.5,
                                mode='crop',
                                spi=None,
                                void_label=0,
                                color_mode='RGB',
                                gtloader=None):
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
        parameters[counter] = [f, g, size, asarray,
                               scale, center, strides,
                               mode, spi, void_label,
                               color_mode, gtloader]
        counter += 1
        if counter == multiprocess:
            samples_and_labels = worker.map(load_image_multiprocess_worker,
                                            parameters)
            counter = 0
            for sample, label in samples_and_labels:
                if len(sample) > 0:
                    samples.extend(sample)
                    if label is not None:
                        labels.extend(label)
    if counter != 0:
        samples_and_labels = worker.map(load_image_multiprocess_worker,
                                        parameters[:counter])
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
                          color_mode='RGB',
                          gtloader=None):
    """ load image in single process
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
            sample, _ = load_image(f, None, size, asarray,
                                   scale, center, strides,
                                   mode, spi, void_label,
                                   color_mode, gtloader)
            if len(sample) > 0:
                samples.extend(sample)
    else:
        assert len(filelist) == len(gtlist)
        for f, g in zip(filelist, gtlist):
            if not os.path.isfile(g):
                logging.warning('ground truth {} not exist, file {} ignored'
                                .format(g, f))
                continue
            sample, label = load_image(f, g, size, asarray,
                                       scale, center, strides,
                                       mode, spi, void_label,
                                       color_mode, gtloader)
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
                   color_mode='RGB',
                   gtloader=None):
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
    if gtloader is None:
        def _loader(x):
            return sm.imread(x, mode=color_mode)
        gtloader = _loader
    if multiprocess != 1:
        return load_from_list_multiprocess(filelist, gtlist, multiprocess,
                                           size, asarray, scale, center,
                                           strides, mode, spi, void_label,
                                           color_mode, gtloader)
    else:
        return load_from_list_simple(filelist, gtlist, size, asarray, scale,
                                     center, strides, mode, spi, void_label,
                                     color_mode, gtloader)


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
                   color_mode='RGB',
                   gtloader=None):
    filelist, gtlist = load_filename(listname, num, basepath, sep, namefilter)
    return load_from_list(filelist, gtlist,
                          size, asarray, scale,
                          center, strides, mode,
                          spi, void_label, multiprocess,
                          color_mode, gtloader)


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
                  color_mode='RGB',
                  gtloader=None):
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
    filelist, gtlist = load_filename_from_dir(imagedir,
                                              gtdir,
                                              gtext,
                                              num,
                                              namefilter)

    return load_from_list(filelist, gtlist,
                          size, asarray, scale,
                          center, strides, mode,
                          spi, void_label, multiprocess,
                          color_mode, gtloader)


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
    train_samples, train_labels = None, None
    valid_samples, valid_labels = None, None
    test_samples , test_labels  = None, None
    eval_samples , eval_labels  = None, None
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
                        yield inputs
                    else: # train / valid phase
                        yield inputs, targets
    return _generate, \
           map(_nsamples,
               [train_samples, valid_samples, test_samples, eval_samples])


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
              gtloader=None,
              num=None,
              namefilter=None,
              nclass=None,
              onehot=False):
    """ generator to generate batch of images
        Attributes
        ----------
            imagedir : string
                       source image directory
            batch_size : int
                         size to generate images for once
            gtdir : string
                    ground truth image directory
            gtext : string
                    ground truth image filename suffix
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
    filelist, gtlist = load_filename_from_dir(imagedir, gtdir, gtext,
                                              num, namefilter)
    filelist = np.asarray(filelist, dtype=np.string_)
    if gtlist is not None:
        gtlist = np.asarray(gtlist, dtype=np.string_)
    length = len(filelist)
    index = np.arange(length, dtype=np.int32)
    iterations = max(int(length / batch_size), 1)
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
                                      void_label, multiprocess,
                                      color_mode, gtloader)
                if onehot:
                    if y is not None:
                        y = one_hot(y, nclass)
                yield [x, y], iteration
    return _generate(), iterations
