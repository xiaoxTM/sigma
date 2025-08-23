import glob
from typing import List, Dict

import numpy as np
import re
import cv2
import typing
import os
from PIL import Image, ImageFile

import zipfile
from sigma.core.fontstyles import colors
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import OpenEXR
    import Imath
    import imageio


    def exr2hdr(exrpath: str) -> np.ndarray:
        File = OpenEXR.InputFile(exrpath)
        PixType = Imath.PixelType(Imath.PixelType.FLOAT)
        DW = File.header()['dataWindow']
        CNum = len(File.header()['channels'].keys())
        if (CNum > 1):
            Channels = ['R', 'G', 'B']
            CNum = 3
        else:
            Channels = ['G']
        Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
        Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
        hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
        if (CNum == 1):
            hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        else:
            hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
            hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
            hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
        return hdr


    def load_exr(filename: str) -> np.ndarray:
        hdr = exr2hdr(filename)
        _, _, c = hdr.shape
        if c == 1:
            hdr = np.squeeze(hdr)
        return hdr

except ImportError as ie:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


    def load_exr(filename: str):
        exr = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
        if exr.shape[2] == 3:
            exr = exr[..., ::-1]
        return exr


def load_flo(filename: str) -> np.ndarray:
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, logger.fatal('Magic number incorrect. Invalid .flo file: {}'.format(filename))
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # Reshape data into 3D array (columns, rows, bands)
        # The reshape here is for visualization, the original code is (w,h,2)
        return np.resize(data, (int(h), int(w), 2))


def save_flo(filename: str, uv: np.ndarray, v: np.ndarray = None) -> None:
    # def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    # known as TAG_CHAR
    f.write(np.array([202021.25], np.float32))
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def load_pfm(filename: str) -> typing.Tuple[np.ndarray, float]:
    file = open(filename, 'r', encoding="ISO-8859-1")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()
    return np.reshape(data, shape), scale


def load_others(filename: str, mode: typing.Optional[str] = 'RGB', inverse: typing.Optional[bool] = None) -> np.ndarray:
    try:
        img = np.asarray(Image.open(filename).convert(mode))
        if inverse:
            img = img[..., ::-1]
        return img
    except IOError as ioe:
        logger.fatal('error when loading {}'.format(filename))
        logger.fatal(ioe)


def save_others(filename: str, array: np.ndarray) -> None:
    img = Image.fromarray(array.astype(np.uint8))
    img.save(filename)


def load_disp(filename: str, scale: float = 1.0) -> np.ndarray:
    try:
        assert os.path.exists(filename), 'file: {} not exists'.format(filename)
        if filename.endswith('.exr'):
            return load_exr(filename)[..., 1]
        elif filename.endswith('.pfm'):
            return load_pfm(filename)[0]
        elif filename.endswith('.npy'):
            return np.load(filename)
        elif filename.endswith('.npz'):
            disp_file = np.load(filename)
            disp = disp_file['disp']
            disp_file.close()
            return disp
        elif filename.endswith('.flo'):
            return load_flo(filename).astype(np.float32)
        elif filename.endswith('.png'):
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
        else:
            raise TypeError('disparity type `{}` not support yet: {}'.format(filename.rsplit('.')[-1],
                                                                             filename))
    except Exception as ioe:
        logger.fatal('error when loading {}'.format(filename))
        logger.fatal(ioe)

def load_depth(filename: str, scale: float = 0.00025) -> np.ndarray:
    return load_disp(filename, scale)

def load_hist(filename: str, scale: float=1.0) -> np.ndarray:
    assert os.path.exists(filename), 'file: {} not exists'.format(filename)
    if filename.endswith('.png'):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED).reshape(96, 240, -1).astype(np.float32)[..., :254]
    return load_histogram(filename).astype(np.float32)[...,:254]

def load_samples(filename: str) -> typing.List[typing.Dict[str, str]]:
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    samples = []
    split = None
    if filename.endswith('csv'):
        split = ','
    for idx, line in enumerate(lines):
        splits = line.split(split)
        if len(splits) == 2:
            samples.append({'left': splits[0], 'right': splits[1]})
        elif len(splits) == 3:
            samples.append({'left': splits[0], 'right': splits[1], 'disp': splits[2]})
        elif len(splits) == 4:
            samples.append({'left': splits[0], 'right': splits[1], 'disp': splits[2], 'seg': splits[3], 'mask': splits[3]})
        # elif len(splits) == 5:
        #     samples.append({'left': splits[0], 'right': splits[1], 'disp': splits[2], 'seg': splits[3], 'mask': splits[4]})
        else:
            raise ValueError(f'at {idx} line: {line}')
    return samples


def load_dataset(pred_path: str, true_path: str, image_path: str = None,
                 mask_path: str = None) -> list[dict[str, str | None]]:
    samples = []
    if os.path.isdir(pred_path):
        pred_samples = os.listdir(pred_path)
        for pred_sample in pred_samples:
            filename = os.path.basename(pred_sample)
            name_without_ext = os.path.splitext(filename)[0]
            image_sample = None
            true_sample = None
            mask_sample = None
            if true_path is not None:
                basename = os.path.splitext(filename)[0]
                candidates = glob.glob(os.path.join(true_path, basename+'*'))
                if len(candidates) == 0:
                    true_sample = None
                    logger.warning('true sample for {} not found in {}'.format(filename, os.path.join(true_path, basename)))
                elif len(candidates) == 1:
                    true_sample = candidates[0]
                else:
                    true_sample = None
                    logger.warning('more than one true samples found for {} in {}'.format(filename,
                                                                                        os.path.join(true_path,
                                                                                                    basename)))
            if image_path is not None:
                image_sample = os.path.join(image_path, name_without_ext+'.jpg')
                if not os.path.exists(image_sample):
                    image_sample = os.path.join(image_path, name_without_ext+'.png')
                    if not os.path.exists(image_sample):
                        logger.warning('image sample {} not exists'.format(image_sample))
                        image_sample = None
            if mask_path is not None:
                mask_sample = os.path.join(mask_path, name_without_ext+'.png')
                if not os.path.exists(mask_sample):
                    mask_sample = None
                    logger.warning('mask sample {} not exists'.format(mask_sample))
            samples.append({'pred': os.path.join(pred_path, pred_sample), 'true': true_sample, 'image': image_sample,
                            'mask': mask_sample})
    elif os.path.isfile(pred_path):
        with open(pred_path, 'r') as fp:
            pred_samples = fp.read().splitlines()
        with open(true_path, 'r') as ft:
            true_samples = ft.read().splitlines()
        image_samples = None
        if image_path is not None:
            with open(image_path, 'r') as fi:
                image_samples = fi.read().splitlines()
        mask_samples = None
        if mask_path is not None:
            with open(mask_path, 'r') as fm:
                mask_samples = fm.read().splitlines()
        for idx, (pred_sample, true_sample) in enumerate(zip(pred_samples, true_samples)):
            image_sample = None
            mask_sample = None
            if image_samples is not None:
                image_sample = image_samples[idx]
            if mask_samples is not None:
                mask_sample = mask_samples[idx]
            samples.append({'pred': pred_sample, 'true': true_sample, 'image': image_sample, 'mask': mask_sample})
    else:
        raise ValueError('pred_path {} is not a file or directory'.format(pred_path))
    return samples


def archive(filename: str,
            path: str,
            pattern: str = None,
            dir_exclusion: typing.List[str] = None,
            file_exclusion: typing.List[str] = None) -> None:
    assert os.path.isdir(path), 'path {} either is not directory or not exists'.format(path)
    if dir_exclusion is None:
        dir_exclusion = []
    if file_exclusion is None:
        file_exclusion = []

    if pattern is not None:
        pattern = re.compile(pattern)

    zf = zipfile.ZipFile(filename, 'w')
    for root, dirs, files in os.walk(path):
        for d in dir_exclusion:
            if d in dirs:
                dirs.remove(d)
        if pattern is not None:
            for idx in range(len(dirs) - 1, -1, -1):
                if pattern.match(dirs[idx]) is not None:
                    dirs.remove(dirs[idx])
        zf.write(root)
        for f in files:
            if f not in file_exclusion and (pattern is None or pattern.match(f) is None):
                zf.write(os.path.join(root, f))
    zf.close()

def convert_raw10_to_swapped_histogram_simd(raw10_data: np.ndarray,
                                            shape: list[int] | tuple[int, int, int] | np.ndarray = [96, 240, 256],
                                            mapping: list[int] | tuple | np.ndarray = [0, 2, 4, 6, 1, 3, 5, 7]) -> np.ndarray:
    # Dimensions: x=240, y=96
    height, width, num_bins = shape
    # 2 x uint8 => uint16
    histogram_data = np.zeros((height, width, num_bins), dtype=np.uint16)

    # Process each index in the raw10_data
    for idx in range(0, len(raw10_data) // 5):
        input_idx = idx * 5
        pix_idx = idx * 4 // num_bins
        pix_bin_idx = idx * 4 % num_bins
        row_idx = pix_idx // width
        col_idx = pix_idx % width

        # Row mapping
        mapped_row_idx = mapping[row_idx % 8]
        group_idx = row_idx // 8

        # Adjust the output index to account for the mapped row
        adjusted_output_idx = ((mapped_row_idx + (group_idx * 8)) * width + col_idx) * num_bins + pix_bin_idx

        # Extract the byte data
        byte_data = raw10_data[input_idx:input_idx+16]
        ushort_data = np.array([
            (((int(byte_data[1]) << 2) | ((int(byte_data[4]) >> 2) & 0x03)) & 0x3FF),
            (((int(byte_data[0]) << 2) | (int(byte_data[4]) & 0x03)) & 0x3FF),
            (((int(byte_data[3]) << 2) | ((int(byte_data[4]) >> 6) & 0x03)) & 0x3FF),
            (((int(byte_data[2]) << 2) | ((int(byte_data[4]) >> 4) & 0x03)) & 0x3FF),
        ], dtype=np.uint16)

        # Store the result into histogram_data
        histogram_data.flat[adjusted_output_idx:adjusted_output_idx+4] = ushort_data

    return histogram_data


def convert_raw8_to_swapped_histogram_simd(raw8_data: np.ndarray,
                                           shape: list[int] | tuple[int, int, int] | np.ndarray = [96, 240, 256],
                                           mapping: list | tuple | np.ndarray = [0, 2, 4, 6, 1, 3, 5, 7]) -> np.ndarray:
    # Dimensions: x=240, y=96
    height, width, num_bins = shape
    # 2 x uint8 => uint16
    histogram_data = np.zeros((height, width, num_bins), dtype=np.uint16)

    # Process each index in the raw10_data
    for idx in range(0, len(raw8_data) // num_bins):
        pix_idx = idx
        row_idx = pix_idx // width
        col_idx = pix_idx % width

        # Row mapping
        mapped_row_idx = mapping[row_idx % 8]
        group_idx = row_idx // 8

        # Adjust the output index to account for the mapped row
        adjusted_output_idx = ((mapped_row_idx + (group_idx * 8)) * width + col_idx) * num_bins

        # Store the result into histogram_data
        histogram_data.flat[adjusted_output_idx:adjusted_output_idx+num_bins] = raw8_data[idx*num_bins:idx*num_bins+num_bins]
        # 将256个数据按照位置奇偶调换
        for i in range(0, num_bins, 2):
            histogram_data.flat[adjusted_output_idx+i], histogram_data.flat[adjusted_output_idx+i+1] = histogram_data.flat[adjusted_output_idx+i+1], histogram_data.flat[adjusted_output_idx+i]

    return histogram_data


def load_histogram_from_raw(filename: str,
                            format_type: str = 'auto',
                            shape: list[int] | tuple[int, int, int] | np.ndarray = [96, 240, 256],
                            mapping: list | tuple | np.ndarray = [0, 2, 4, 6, 1, 3, 5, 7]) -> np.ndarray:
    assert format_type in ['raw8', 'raw10', 'auto'], colors.red('invalid format type: {}'.format(format_type))
    filesize = os.path.getsize(filename) // (1024 ** 2)

    if format_type in ['auto']:
        format_type = 'raw8' if abs(filesize - 11) < 2 else 'raw10'
        logger.debug(colors.blue('format type auto detected as {}'.format(format_type)))

    if format_type in ['raw10']:
        if abs(filesize - 11) < 2:
            logger.warning(colors.yellow('detected raw8 format but format_type is given as raw10'))
        # raw10_data should be a numpy array with 11796480 bytes (i.e., 240 * 96 * 256 * 2 / 2 for raw10)
        # 读取二进制文件为np.uint8
        raw10_data = np.fromfile(filename, dtype=np.uint8)
        histogram_data = convert_raw10_to_swapped_histogram_simd(raw10_data, shape, mapping)
    else:
        if abs(filesize - 7) < 2:
            logger.warning(colors.yellow('detected raw10 format but format_type is given as raw8'))
        raw8_data = np.fromfile(filename, dtype=np.uint16)
        histogram_data = convert_raw8_to_swapped_histogram_simd(raw8_data, shape, mapping)
    return histogram_data


def load_histogram(filename: str, **kwargs) -> np.ndarray:
    ext = os.path.splitext(filename)[1]
    if ext == '.raw':
        args = {k:v for k, v in kwargs.items() if k in ['format_type', 'shape', 'mapping']}
        data = load_histogram_from_raw(filename, **args)
    elif ext == '.npz':
        try:
            hist_file = np.load(filename)
            hist = hist_file['hist']
            hist_file.close()
            return hist
        except Exception as e:
            print(e)
            raise ValueError(f'loading {filename} failed')
    elif ext == '.npy':
        data = np.load(filename)
    elif ext == '.txt':
        data = np.loadtxt(filename, skiprows=1, dtype=np.int32).reshape(96, 240, 256)[..., :254]
    else:
        raise ValueError('invalid file format: {} for {}'.format(colors.red(ext), filename))
    # data: [H, W, 254]
    # rorate along the depth axis [254]
    if kwargs.get('flip', False):
        data = np.flip(data, axis=(0, 1))
    return data


def print_histogram(h, w, hist):
    print(h, w, hist[h, w, 64])


if __name__ == '__main__':
    filename = 'frame_00_tx_40.6250_rx_42.5.raw'
    histogram_data = load_histogram(filename)
    print(histogram_data.shape)  # Should output (96, 240, 256)

    # %% 测试数据
    print_histogram(30, 27, histogram_data)
