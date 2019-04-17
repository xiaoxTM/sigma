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
import multiprocessing as mp
from scipy import misc as sm
import os
import os.path
import logging
from sigma.helpers import nput
from .. import utils

def save_image(filename, image):
    sm.imsave(filename, image)


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


###############################################################################
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


###############################################################################
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
    filelist, gtlist = utils.load_filename(listname, num, basepath, sep, namefilter)
    return load_from_list(filelist, gtlist,
                          size, asarray, scale,
                          center, strides, mode,
                          spi, void_label, multiprocess,
                          color_mode, gtloader)


###############################################################################
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
    filelist, gtlist = utils.load_filename_from_dir(imagedir,
                                                    gtdir,
                                                    gtext,
                                                    num,
                                                    namefilter)

    return load_from_list(filelist, gtlist,
                          size, asarray, scale,
                          center, strides, mode,
                          spi, void_label, multiprocess,
                          color_mode, gtloader)
