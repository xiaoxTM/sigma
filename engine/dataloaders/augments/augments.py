from __future__ import division
import torch
import numpy as np
from PIL import Image
import typing
from sigma.engine.misc import misc


class Compose(object):
    def __init__(self, transforms: typing.List[typing.Callable]):
        self.transforms = transforms

    def __call__(self,
                 sample: typing.Dict[str, typing.Union[np.ndarray, torch.Tensor, Image.Image]]) -> (
            typing.Dict)[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class LeftTopCrop(object):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:
        height, _ = list(sample.values())[0].shape
        start_point = [height - self.height, 0]

        for key in sample.keys():
            sample[key] = misc.fit_shape(sample[key], self.height, self.width, start_point=start_point,
                                         ftype='crop_or_pad', mode='constant', constant_values=0)
        return sample


class CenterCrop(object):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:

        origin_height, origin_width = sample[list(sample.keys())[0]].shape[:2]
        start_point = [(origin_height - self.height) // 2, (origin_width - self.width) // 2]
        end_point = [(origin_height + self.height) // 2, (origin_width + self.width) // 2]

        for key in sample.keys():
            sample[key] = misc.fit_shape(sample[key], self.height, self.width, ftype='crop_or_pad',
                                         start_point=start_point, end_point=end_point, mode='constant',
                                         constant_values=0)

        return sample


class TopCenterCrop(object):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:
        origin_height, origin_width = list(sample.values())[0].shape[:2]
        start_point = [(origin_height - self.height), (origin_width - self.width) // 2]
        end_point = [origin_height, (origin_width + self.width) // 2]

        for key in sample.keys():
            sample[key] = misc.fit_shape(sample[key], self.height, self.width, ftype='crop_or_pad',
                                         start_point=start_point, end_point=end_point, mode='constant',
                                         constant_values=0)
        return sample


class RandomCrop(object):
    def __init__(self, height: int, width: int, validate: bool = False):
        self.img_height = height
        self.img_width = width
        self.validate = validate

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:
        ori_height, ori_width = sample['left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            for key in self.sample.keys():
                sample[key] = np.lib.pad(sample[key],
                                         ((0, top_pad), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validation, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            for key in sample.keys():
                sample[key] = self.crop(sample[key])

        return sample

    def crop(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        return img[self.offset_y:self.offset_y + self.img_height, self.offset_x:self.offset_x + self.img_width]


class RandomHorizonFlip(object):
    """Randomly horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:
        if np.random.random() < self.p:
            for key in sample.keys():
                sample[key] = np.copy(np.fliplr(sample[key]))
        return sample


class RandomNoise(object):
    def __init__(self, p: float = 0.5, mu: float = 0.1, sigma: float = 0.32, keys=None):
        if keys is None:
            keys = ['left', 'right']
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.keys = keys

    def __call__(self, sample: typing.Dict[str, np.ndarray]) -> typing.Dict[str, np.ndarray]:
        if 'left' in self.keys:
            left = sample['left']
            if np.random.random() < self.p:
                left = noise.add_poisson_noise(left)
            if np.random.random() < self.p:
                left = noise.add_dynamic_range_noise(left)
            sample['left'] = left
        if 'right' in self.keys and np.random.random() < self.p:
            right = sample['right']
            if np.random.random() < self.p:
                right = noise.add_poisson_noise(right)
            if np.random.random() < self.p:
                right = noise.add_dynamic_range_noise(right)
            sample['right'] = right
        return sample



# if __name__ == '__main__':
#     import copy
#     dist = np.random.randint(0, 10, size=(2, 2, 10))
#     dist1 = copy.deepcopy(dist)
#     rs = RandomShift(p=1.0)
#     data = {'hist': dist}
#     import matplotlib.pyplot as plt
#     plt.subplot(221)
#     plt.plot(dist1[0,0], c='r')
#     plt.subplot(222)
#     plt.plot(dist1[0,1], c='r')
#     plt.subplot(223)
#     plt.plot(dist1[1,0], c='r')
#     plt.subplot(224)
#     plt.plot(dist1[1,1], c='r')
#     rs(data)
#     plt.subplot(221)
#     plt.plot(data['hist'][0, 0], 'b-.')
#     plt.subplot(222)
#     plt.plot(data['hist'][0, 1], 'b-.')
#     plt.subplot(223)
#     plt.plot(data['hist'][1, 0], 'b-.')
#     plt.subplot(224)
#     plt.plot(data['hist'][1, 1], 'b-.')
#     plt.show()
