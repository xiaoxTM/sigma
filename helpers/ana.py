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

import scipy.misc as sm
import multiprocessing as mp
import numpy as np
import os
import os.path
from .nput import load_filename

def histogram_multiprocess_worker(params):
    name, nclass = params
    distributes = np.zeros(nclass)
    label = sm.imread(name)
    for nc in range(nclass):
        distributes[nc] += np.sum(label == nc)
    return distributes


def histogram_from_list_multiprocess(listname, nclass, basepath):
    assert isinstance(listname, (list, tuple))
    distributes = np.zeros(nclass)
    counter = 0
    if multiprocess is None or multiprocess < 1:
        multiprocess = mp.cpu_count()
    parameters = [None] * multiprocess
    worker = mp.Pool(processes=multiprocess)
    for name in listname:
        if basepath is not None:
            name = os.path.join(basepath, name)
        parameters[counter] = [name, nclass]
        counter += 1
        if counter == multiprocess:
            distribute = worker.map(histogram_multiprocess_worker, parameters)
            counter = 0
            for distr in distribute:
                distributes += distr
    if counter != 0:
        distribute = worker.map(histogram_multiprocess_worker,
                                parameters[:counter])
        for distr in distribute:
            distributes += distr
    worker.close()
    return distributes, distributes / np.sum(distributes)


def histogram_from_list_simple(listname, nclass, basepath):
    assert isinstance(listname, (list, tuple))
    distributes = np.zeros(nclass)
    for name in listname:
        filename = name
        if basepath is not None:
            filename = os.path.join(basepath, filename)
        label = sm.imread(filename)
        for nc in range(nclass):
            distributes[nc] += np.sum(label == nc)
    return distributes, distributes / np.sum(distributes)


def histogram_from_list(listname, nclass, basepath,
                        multiprocess=1):
    if multiprocess != 1:
        return histogram_from_list_multiprocess(listname, nclass, basepath)
    else:
        return histogram_from_list_simple(listname, nclass, basepath)


def histogram_from_file(listname, nclass,
                        basepath=None,
                        sep=' ',
                        multiprocess=1,
                        namefilter=None):
    _, gtlist = load_filename(listname, None, basepath, sep, namefilter)
    assert gtlist is not None, 'cannot calculate histogram ' \
                               'if no ground truth file given'
    return histogram_from_list(gtlist, nclass, basepath, multiprocess)


def histogram_from_dir(dirname, nclass, multiprocess, namefilter=None):
    listname = []
    if namefilter is None:
        namefilter = lambda x: True
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if namefilter(os.path.join(root, f)):
                listname.append(os.path.join(root, f))
    return histogram_from_list(listname, nclass, None, multiprocess)


def estimate_class_weights_from_file(listname, nclass,
                                     basepath=None,
                                     normalize=True,
                                     sep=' ',
                                     multiprocess=1,
                                     namefilter=None):
    distributes = np.zeros(nclass)
    if isinstance(listname, str):
        listname = [listname]
    assert isinstance(listname, (tuple, list))
    for l in listname:
        _, distribute = histogram_from_file(l, nclass,
                                            basepath,
                                            sep, multiprocess,
                                            namefilter)
        distributes += distribute
    if normalize:
        distrexp = np.exp(-distributes)
        return distrexp / np.sum(distrexp)
    else:
        return np.exp(-distributes)


def estimate_class_weights_from_dir(dirname, nclass,
                                    normalize=True,
                                    multiprocess=1,
                                    namefilter=None):
    distributes = np.zeros(nclass)
    if isinstance(dirname, str):
        dirname = [dirname]
    assert isinstance(dirname, (tuple, list))
    for d in dirname:
        _, distribute = histogram_from_dir(d, nclass, multiprocess, namefilter)
        distributes += distribute
    if normalize:
        distrexp = np.exp(-distributes)
        return distrexp / np.sum(distrexp)
    else:
        return np.exp(-distributes)


def distribute_multiprocess_worker(params):
    filename, gtname, nclass = params
    image = sm.imread(filename)
    label = sm.imread(gtname)
    distributes = np.zeros([3, nclass, 256])
    for channel in range(3):
        single = image[:, :, channel]
        for nc in range(nclass):
            position = np.where(label == nc)
            hist, _ = np.histogram(single[position], np.arange(257))
            distributes[channel, nc, :] += hist
    return distributes


def distribute_from_list_multiprocess(filename, gtname, nclass,
                                      basepath, multiprocess):
    assert isinstance(filename, (list, tuple))
    assert isinstance(gtname, (list, tuple))
    distributes = np.zeros([3, nclass, 256])
    counter = 0
    if multiprocess is None or multiprocess < 1:
        multiprocess = mp.cpu_count()
    parameters = [None] * multiprocess
    worker = mp.Pool(processes=multiprocess)
    for f, g in zip(filename, gtname):
        if basepath is not None:
            f = os.path.join(basepath, f)
            g = os.path.join(basepath, g)
        parameters[counter] = [f, g, nclass]
        counter += 1
        if counter == multiprocess:
            distribute = worker.map(distribute_multiprocess_worker, parameters)
            counter = 0
            for distr in distribute:
                distributes += distr
    if counter != 0:
        distribute = worker.map(distribute_multiprocess_worker,
                                parameters[:counter])
        for distr in distribute:
            distributes += distr
    worker.close()
    return distributes


def distribute_from_list_simple(filename, gtname, nclass, basepath):
    assert isinstance(filename, (list, tuple))
    assert isinstance(gtname, (list, tuple))
    distributes = np.zeros([3, nclass, 256])
    for f, g in zip(filename, gtname):
        if basepath is not None:
            f = os.path.join(basepath, f)
            g = os.path.join(basepath, g)
        image = sm.imread(f)
        label = sm.imread(g)
        for channel in range(3):
            single = image[:, :, channel]
            for nc in range(nclass):
                position = np.where(label == nc)
                hist, _ = np.histogram(single[position], np.arange(257))
                distributes[channel, nc, :] += hist
    return distributes


def distribute_from_list(filename, gtname, nclass, basepath, multiprocess=1):
    if multiprocess != 1:
        return distribute_from_list_multiprocess(filename, gtname, nclass,
                                                 basepath, multiprocess)
    else:
        return distribute_from_list_simple(filename, gtname, nclass, basepath)


def distribute_from_file(listname, nclass,
                         basepath=None,
                         sep=' ',
                         multiprocess=1,
                         namefilter=None):
    filelist, gtlist = load_filename(listname, None,
                                     basepath, sep, namefilter)
    assert filelist is not None, 'cannot calculate distribution ' \
                                 'if no image file given'
    assert gtlist is not None, 'cannot calculate distribution ' \
                               'if no ground truth file given'
    return distribute_from_list(filelist, gtlist, nclass,
                                basepath, multiprocess)


if __name__ == '__main__':
    print(distribute_from_file('/home/xiaox/studio/db/camvid/segnet/test.txt',
                               12,
                               basepath='/home/xiaox/studio/db/camvid/segnet/'))
