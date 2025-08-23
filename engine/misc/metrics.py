import torch
from . import transformation, ios, transformation as trans, pcio, misc
import numpy as np
# import itertools
from typing import List, Dict, Union, Tuple
from collections import OrderedDict
import termtables as tt
import typing
import copy
import logging
import cv2
import os.path

import matplotlib.pyplot as plt

from sigma.core.fontstyles import colors

import warnings
warnings.simplefilter('once')

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def evaluate_depth_errors(pred: np.ndarray,
                          true: np.ndarray,
                          ranges: Union[List[float], np.ndarray],
                          trim: Tuple[float] = (0, 100000),
                          image_mean: bool = True) -> List[Dict[str, float]]:
    # pred: depth prediction
    # true: depth ground truth
    absolute = OrderedDict()
    relative = OrderedDict()
    num_pixels = OrderedDict()

    if image_mean:
        call = np.nanmean
    else:
        call = np.nansum

    prev = ranges[0]

    trim_mask = (true > trim[0]) & (true < trim[1]) & (pred > 0)
    cliped_pred = copy.deepcopy(pred)
    # cliped_pred[trim_mask & (pred < trim[0])] = trim[0]
    # cliped_pred[trim_mask & (pred > trim[1])] = trim[1]
    for ran in ranges[1:]:
        key = '{}-{}'.format(prev, ran)
        absolute.update({key: 0})
        relative.update({key: 0})
        num_pixels.update({key: 0})
        cmask = np.logical_and(true > prev, true <= ran)
        mask = np.logical_and(cmask, trim_mask)
        if mask.sum() > 0:
            mask_pred = cliped_pred[mask]
            mask_true = true[mask]
            acc = abs(mask_pred - mask_true)
            per = call(acc / mask_true, dtype=np.float64) + relative.get(key, 0.0)
            acc = call(acc, dtype=np.float64) + absolute.get(key, 0.0)
            num_pixel = num_pixels.get(key, 0.0) + mask.sum()
            absolute.update({key: acc})
            relative.update({key: per})
            num_pixels.update({key: num_pixel})
        prev = ran
    return [absolute, relative, num_pixels]


def evaluate_errors(disparity: np.ndarray, truth: np.ndarray, baseline: float, focus: float,
                    ranges: Union[List[float], np.ndarray],
                    trim: Tuple[float] = (0, 100000),
                    image_mean: bool = True) -> List[Dict[str, float]]:
    depth = transformation.disparity_to_depth(disparity, baseline, focus)
    depth_truth = transformation.disparity_to_depth(truth, baseline, focus)
    return evaluate_depth_errors(depth, depth_truth, ranges, trim, image_mean)


def evaluate_from_list(filenames: List[List[str]],
                       baseline: float,
                       focus: float,
                       scale: float,
                       ranges: Union[List[float], np.ndarray],
                       trim: Tuple[float] = (0, 100000),
                       image_mean: bool = False) -> List[Dict[str, List[List[float]]]]:
    absolute = OrderedDict()
    relative = OrderedDict()
    num_pixels = OrderedDict()

    for filename in filenames:
        name_pred, name_true = filename
        pred = ios.load_disp(name_pred, 1.0)
        true = ios.load_disp(name_true, scale)
        ab, re, num = evaluate_errors(pred, true, baseline, focus, ranges, trim, image_mean)

        for key in ab.keys():
            absl = ab[key] + absolute.get(key, 0.0)
            absolute.update({key: absl})
            rel = re[key] + relative.get(key, 0.0)
            relative.update({key: rel})
            num_pixel = num[key] + num_pixels.get(key, 0.0)
            num_pixels.update({key: num_pixel})

    return [absolute, relative, num_pixels]


def evaluate_accuracy(stats: List[Dict[str, List[List[float]]]]) -> \
        List[Union[Dict[str, float], Dict[str, Dict[str, float]]]]:
    abl, rel, nums = stats

    # all the pixel average
    mean_stat = OrderedDict()
    # nums = OrderedDict()

    # range-based average
    mean_abs_range_stat = OrderedDict()
    # for absolute accuracy

    total_nums = np.sum(list(nums.values()))

    global_mean_abs = 0.0

    for k, v in abl.items():
        # v: List of List[float]
        # flatten: List[float]
        global_mean_abs += v / total_nums
        mean_abs_range_stat.update({k: v / nums[k]})
    mean_stat.update({'abs': global_mean_abs})

    # range-based average
    mean_rel_range_stat = OrderedDict()
    # for relative accuracy

    global_mean_rel = 0.0
    for k, v in rel.items():
        # v: List of List[float]
        # flatten: List[float]
        global_mean_rel += v / total_nums
        mean_rel_range_stat.update({k: v / nums[k]})
    mean_stat.update({'rel': global_mean_rel})

    mean_stat.update({'num': np.mean(list(nums.values()))})

    return [mean_stat, {'abs': mean_abs_range_stat, 'rel': mean_rel_range_stat}]
