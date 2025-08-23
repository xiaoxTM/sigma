#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import cv2
import numpy as np
from . import ios
import typing
import logging
from sigma.engine.dataloaders import augments as augs
from sigma.core.fontstyles import colors

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def load_from_dir(path: str, ext: str = 'npz') -> typing.List[typing.List[str]]:
    logger.info('loading data from {}'.format(colors.blue(path)))
    #lists = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
    #            f.rsplit('.')[-1] in ['png', 'raw', 'npz', 'npy', 'txt']]
    data = []
    for hist in os.listdir(os.path.join(path, 'hist')):
        datum = [os.path.join(path, 'hist', hist)]
        depth = os.path.join(path, 'depth', hist)
        if os.path.exists(depth): # depth GT file exists
            datum.append(depth)
            image = os.path.join(path, 'image', hist.replace('.png', '.jpg'))
            if os.path.exists(image):
                datum.append(image)
                pc = os.path.join(path, 'pointcloud', hist.replace('.png', '.ply'))
                if os.path.exists(pc):
                    datum.append(pc)
        data.append(datum)
    return data


class Stream:
    def __init__(self, source: typing.Union[int, str], **kwargs):
        if isinstance(source, str):
            if source.rsplit('.', 1)[-1] in ['list', 'txt', 'csv']:
                assert os.path.isfile(source), 'file {} not exists'.format(source)
                with open(source, 'r') as fp:
                    lists = fp.read().splitlines()
                root = kwargs.get('root', None)
                split = ',' if source.endswith('csv') else None
                if root is not None:
                    lists = [[os.path.join(root, l) for l in ls.split(split)] for ls in lists]
                self.data = []
                for ls in lists:
                    all_exists = True
                    for l in ls:
                        if not os.path.exists(l):
                            all_exists = False
                            logger.info('file not exists: {}'.format(colors.red(l)))
                    if all_exists:
                        self.data.append(ls)
                self.num_data = len(self.data)
            elif source.rsplit('.', 1)[-1] in ['avi', 'mp4', 'webm']:
                assert os.path.isfile(source), 'file {} not exists'.format(source)
                self.data = cv2.VideoCapture(source)
                assert self.data.isOpened, 'open video file:{} failed'.format(source)
                self.num_data = int(self.data.get(cv2.CAP_PROP_FRAME_COUNT))
            elif os.path.isdir(source):
                self.data = load_from_dir(source, kwargs.get('disparity-file-extension', 'npz'))
                self.num_data = len(self.data) # hist file ONLY
            else:
                raise TypeError('source {} not support'.format(source))
            logger.info('total {} samples found.'.format(colors.blue('{:,}'.format(self.num_data))))
        elif isinstance(source, int):
            # from camera
            self.data = cv2.VideoCapture(source)
            self.num_data = 0
            height = self.data.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = self.data.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.data.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
            self.data.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.augments = augs.get(kwargs.get('augments', []))
        self.num_frame = 0
        self.scale = kwargs.get('scale', 0.00025)
        logger.info('building stream with augments:{}, scale:{}'.format(colors.blue(kwargs.get('augments', [])),
                                                                        colors.blue(self.scale)))

    def __next__(self) -> typing.Union[typing.Tuple[np.ndarray, int],
                                       typing.Tuple[np.ndarray, np.ndarray, str],
                                       typing.Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        return self.next()

    def __iter__(self):
        self.num_frame = 0
        return self

    def __len__(self) -> int:
        return self.num_data

    def next(self) -> typing.Dict[str, np.ndarray]:
        if self.num_data == 0 or self.num_frame < self.num_data:
            if isinstance(self.data, cv2.VideoCapture):
                # if live stream, there will be no augments applied
                self.num_frame += 1
                rval, frame = self.data.read()
                if not rval:
                    logger.fatal(colors.red('reading stream from camera/video failed'))
                return {'stereo': frame, 'name': self.num_frame}
            else:
                sample = self.data[self.num_frame]
                self.num_frame += 1
                ret = {}
                if isinstance(sample, dict):
                    hist = ios.load_hist(sample['hist'])
                    if hist.shape[-1] == 254:
                        hist = np.pad(hist, ((0, 0), (0, 0), (1, 1)), constant_values=(0, 0))
                    ret.update({'hist': hist})
                    if 'depth' in sample.keys():
                        depth = ios.load_depth(sample['depth'])
                        ret.update({'depth': depth})
                    if 'mask' in sample.keys():
                        mask = ios.load_others(sample['mask'], mode='L')
                        ret.update({'mask': mask})
                    else:
                        mask = np.ones_like(hist[:, :, 0], dtype=np.uint8)
                        ret.update({'mask': mask})
                    ret.update({'name': sample['hist']})
                else:
                    if isinstance(sample, str):
                        sample = sample.split()
                    assert len(sample) in [1, 2, 3, 4], 'bad sample length: {}'.format(len(sample))
                    if len(sample) > 0:
                        hist = ios.load_hist(sample[0])
                        if hist.shape[-1] == 254:
                            hist = np.pad(hist, ((0, 0), (0, 0), (1, 1)), constant_values=(0, 0))
                        hist_name = sample[0]
                        ret.update({'hist': hist})
                    if len(sample) > 1:
                        depth_name = sample[1]
                        depth = ios.load_depth(depth_name, self.scale)
                        ret.update({'depth': depth})
                    if len(sample) > 2:
                        image_name = sample[2]
                        image = ios.load_others(image_name)
                        ret.update({'image': image})
                    ret.update({'name': sample[0]})
                    # if 'mask' not in ret.keys():
                    #     ret.update({'mask': np.ones_like(ret['left'][:, :, 0], dtype=np.uint8)})
                ret = self.augments(ret)
                return ret

        else:
            raise StopIteration()
