from contextlib import contextmanager
import typing
import cv2
import copy
import torch
from torch import nn
from torch.nn import functional as func
import torch.distributed as dist
import numpy as np
import open3d as o3d
import os.path
import re
import cv2.fisheye
import cv2 as cv
import configparser

from PIL import Image
from sigma.core.fontstyles import colors

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@contextmanager
def keep_status(model: nn.Module, set_train: bool) -> None:
    training = model.training
    model.train(set_train)
    yield
    model.train(training)


def gpu_memory_info(device: torch.device | int = None) -> tuple[float, float, float]:
    # in MB unit
    if torch.cuda.is_available():
        unit_mb = 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / unit_mb
        reserved = torch.cuda.memory_reserved(device) / unit_mb
        used = torch.cuda.memory_allocated(device) / unit_mb
        return total, reserved, used
    return 0, 0, 0


def memory_info() -> tuple[float, float, float]:
    # in MB unit
    with open('/proc/meminfo', 'r') as f:
        lines = f.readlines()
        total_memory = int(lines[0].split()[1]) / 1024.0
        free_memory = int(lines[1].split()[1]) / 1024.0
        used_memory = total_memory - free_memory
        return total_memory, used_memory, free_memory


def trainable_parameters(model: nn.Module) -> float:
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num / 1e6


def freeze_layers_by_name(model: nn.Module,
                          patterns: typing.Union[typing.List[str], typing.Tuple[str], str]) -> nn.Module:
    if isinstance(patterns, str):
        patterns = [patterns]
    for name, param in model.named_parameters():
        for pattern in patterns:
            if re.search(pattern, name):
                logger.info('freezing {}'.format(name))
                param.requires_grad = False
    return model


def log(fun: typing.Callable, message: str) -> None:
    if dist.is_initialized():
        fun('[{}]{}'.format(colors.green('RANK.{}'.format(dist.get_rank())), message))
    else:
        fun(message)


def process_intrinsics(config: typing.Dict[str, typing.Any]) -> None:
    if config['intrinsics'] is not None or config['param_file'] is not None:
        scale = config['intrinsic_scale']
        if config['param_file'] is not None:
            if os.path.exists(config['param_file']):
                if config['param_file'].endswith('.yaml') or config['param_file'].endswith('.yml'):
                    import cv2
                    data = cv2.FileStorage(config['param_file'], cv2.FILE_STORAGE_READ)
                    keys = data.root().keys()
                    if 'baseline' in keys:  # [shenzhen] data collection device format
                        baseline = -data.getNode('baseline').real()
                        im = data.getNode('camera_mat').mat()
                        intrinsics = [im[0, 0], im[1, 1], im[0, 2], im[1, 2]]
                    elif 'T' in keys: # [suzhou] data collection device format
                        assert 'P1' in keys, 'P1 not found in {}'.format(config['param_file'])
                        baseline = -data.getNode('T').mat()[0, 0]
                        im = data.getNode('P1').mat()
                        intrinsics = [im[0, 0], im[1, 1], im[0, 2], im[1, 2]]
                    else:
                        raise TypeError('unsupported YAML file format:{}'.format(config['param_file']))
                    data.release()
                elif config['param_file'].endswith('.json'):
                    import json
                    with open(config['param_file'], 'r') as f:
                        data = json.load(f)
                    baseline = data['baseline']
                    intrinsics = [data['fx'], data['fy'], data['cx'], data['cy']]
                else:
                    raise TypeError('unsupported param file format:{}'.format(colors.red(
                        config['param_file'].rsplit('.', 1)[-1])))
                if config['baseline'] is None:
                    config['baseline'] = baseline
                if config['intrinsics'] is None:
                    config['intrinsics'] = intrinsics
            else:
                raise FileNotFoundError('param file not found:{}'.format(colors.red(config['param_file'])))
        config['intrinsics'] = np.asarray(config['intrinsics']) * scale
        del config['param_file']
        del config['intrinsic_scale']


def parse_remain_arg(arg: str) -> str:
    assert '=' in arg, 'invalid arg: {}'.format(arg)
    k, v = arg.split('=', 1)
    keys = k.strip().split('.')
    key = ''.join("['{}']".format(ks) for ks in keys)
    if v.startswith('#'):
        value = v[1:].split('#')
        if len(value) == 1:
            return '{}={}'.format(key, value[0])
        return '{}=[{}]'.format(key, ', '.join(value))
    value = v.split('#')
    if len(value) == 1:
        return '{}="{}"'.format(key, value[0])
    return '{}={}'.format(key, ["{}".format(va) for va in value])


def fit_plane(pc: np.ndarray,
              mask: np.ndarray,
              distance_threshold: float = 10,
              ransac_n: int = 3,
              num_iterations: int = 10000) -> typing.Union[list[typing.Any], None]:
    masked_pc = pc[mask, :]
    if len(masked_pc) < ransac_n:
        return None

    pc2d = masked_pc.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc2d)
    # distance_threshold=5, distance_threshold=10，最终选择5即可，尽量保证地面是好的
    # 做地面gt数据稠密化时候选择distance_threshold=10，测试地面波动性时用20
    plane_model, _ = pcd.segment_plane(distance_threshold=distance_threshold,
                                       ransac_n=ransac_n,
                                       num_iterations=num_iterations)

    # # 平面模型参数
    # a, b, c, d = plane_model

    # 平面法向量和理想坐标系法向量的旋转
    normal_floor = np.array(plane_model[:-1])
    normal = np.array([0, 1, 0])
    # cos(normal_angle) = np.dot(normal, normal_floor) / (np.linalg.norm(normal) * np.linalg.norm(normal_floor))
    normal_angle = np.arccos(np.dot(normal, normal_floor) / (np.linalg.norm(normal) * np.linalg.norm(normal_floor)))
    angle_diff = normal_angle * 180 / np.pi

    rotation_axis = np.cross(normal_floor, normal)

    # 通过旋转轴+旋转角度得到旋转向量
    rotation_vector = (rotation_axis / np.linalg.norm(rotation_axis)) * normal_angle
    rotate, _ = cv2.Rodrigues(rotation_vector)

    # 对输入点云做矫正，得到地面高度
    rotated_pc = (rotate @ pc2d.T).T
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_pc)

    _, rotated_pcd_inliers = rotated_pcd.segment_plane(distance_threshold=distance_threshold,
                                                       ransac_n=ransac_n,
                                                       num_iterations=num_iterations * 2)
    # 得到矫正后的地面点云
    floor = np.asarray(rotated_pcd.select_by_index(rotated_pcd_inliers).points)

    mean = np.mean(floor[:, 1])
    std = np.std(floor[:, 1])

    return [rotate, mean, std, angle_diff]


def fit_shape(image: np.ndarray, height: int, width: int, ftype: str,
              start_point: typing.Union[tuple[int, int], list[int], np.ndarray[int]] = None,
              end_point: typing.Union[tuple[int, int], list[int], np.ndarray[int]] = None,
              **kwargs) -> np.ndarray:
    """
    Resizes or crops/pads the given image to the specified height and width based on the specified mode.

    Args:
        image (np.ndarray): The input image to be resized or cropped/padded.
        height (int): The desired height of the output image.
        width (int): The desired width of the output image.
        start_point (Union[tuple[int], list[int], np.ndarray[int]]): The starting point for resize/cropping/padding.
                    in the order of [height, width].
        end_point (Union[tuple[int], list[int], np.ndarray[int]]): The ending point for resize/cropping/padding.
                    in the order of [height, width].
        ftype (str): The fit type for resizing or cropping/padding the image.
                     It should be either 'resize' or 'crop_or_pad'.

    Returns:
        np.ndarray: The resized or cropped/padded image based on the specified mode.
    """
    assert ftype in ['resize', 'crop_or_pad'], 'unsupported mode:{}'.format(ftype)
    if start_point is None:
        start_point = [0, 0]
    if end_point is None:
        end_point = [image.shape[0], image.shape[1]]

    assert start_point[0] < end_point[0] and start_point[1] < end_point[1], (
        'start_point({}) >= end_point({})'.format(start_point, end_point))
    if start_point[0] < 0 or start_point[1] < 0 or end_point[0] > image.shape[0] or end_point[1] > image.shape[1]:  # padding
        h1, h2, w1, w2 = 0, 0, 0, 0
        sp = copy.deepcopy(start_point)
        ep = copy.deepcopy(end_point)
        if sp[0] < 0:
            h1 = -sp[0]
            sp[0] = 0
        if sp[1] < 0:
            w1 = -sp[1]
            sp[1] = 0
        if ep[0] >= image.shape[0]:
            h2 = ep[0] - image.shape[0]
        if ep[1] >= image.shape[1]:
            w2 = ep[1] - image.shape[1]
        ep[0] += (h1 + h2)
        ep[1] += (w1 + w2)

        if image.ndim == 2:
            subimage = np.pad(image, ((h1, h2), (w1, w2)), **kwargs)
        elif image.ndim == 3:
            subimage = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), **kwargs)
        else:
            raise ValueError('Unsupported number of dimensions for image array: {}'.format(image.ndim))
        return subimage[sp[0]:ep[0], sp[1]:ep[1]]
    else:
        subimage = image[start_point[0]:end_point[0], start_point[1]:end_point[1]]
        if subimage.shape[0] == height and subimage.shape[1] == width:  # crop
            return subimage
        elif ftype == 'resize':
            return cv2.resize(subimage, (width, height), **kwargs)
        else:
            raise TypeError('Unsupported ftype: {}'.format(ftype))


def load_checkpoint(model: nn.Module, checkpoint: str, strict_load: bool = True) -> None:
    assert os.path.exists(checkpoint), f'checkpoint[{checkpoint}] not exists: No such file or directory'
    logger.info('loading checkpoint from {} with strict[{}]'.format(colors.blue(checkpoint), colors.yellow(strict_load)))

    match = re.search(model.tag, checkpoint)
    if match is None:
        logger.warning(colors.yellow('name of the weight file not match the model tag: {} <=> {}'.format(
            os.path.basename(checkpoint), model.tag)))
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0, 'despite cuda device available, no cuda device found'
        state = torch.load(checkpoint, map_location='cuda:0')
    else:
        state = torch.load(checkpoint, map_location='cpu')
    if 'weights' in state.keys():
        weights = state['weights']
    elif 'state_dict' in state.keys():
        weights = state['state_dict']
    else:
        weights = state
    # if module in key (DDP model), remove it
    state_dict = {k[7:] if 'module' in k else k: v for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=strict_load)


def build_model(args: typing.Dict) -> nn.Module:
    from src.models import architectures as arch
    model = arch.get(args['model']['name'],
                     mixed_precision=args['mixed_precision'],
                     freeze_batch_norm=args['train']['freeze_batch_norm'],
                     maxv=args.get('maxv', 16.0),
                     **args['model']['params'])
    ckpt = args['infer']['checkpoint']
    if ckpt is None:
        logger.warning(colors.yellow('no checkpoint provided for loading'))
    else:
        load_checkpoint(model, ckpt, strict_load=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model


def load_sample(left: str, right: str) -> typing.Dict[str, np.ndarray]:
    image_left = np.asarray(Image.open(left).convert('RGB')).astype(np.float32)
    image_right = np.asarray(Image.open(right).convert('RGB')).astype(np.float32)
    return {'left': image_left, 'right': image_right}


def preprocess(samples: typing.Dict[str, torch.Tensor]) -> typing.Dict:
    samples = camera_ready(samples)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key in ['hist', 'mask']:
        samples[key] = samples[key].unsqueeze(0).to(device, non_blocking=True)
    return samples


def visualize_map(map_: np.ndarray,
                  mask: np.ndarray = None,
                  color: int | str = 2,
                  min_value: float = None,
                  max_value: float = None,
                  bar_size: list[int] | np.ndarray[int] = None) -> np.ndarray | list[np.ndarray]:
    if mask is None:
        mask = np.ones_like(map_, np.bool_)
    if isinstance(color, int):
        image = np.zeros_like(map_, np.uint8)
        masked_map = map_[mask]
        if min_value is None:
            min_value = masked_map.min()
        if max_value is None:
            max_value = masked_map.max()
        np.clip(masked_map, min_value, max_value, masked_map)
        image[mask] = ((masked_map - min_value) * 255 / (max_value - min_value)).astype(np.uint8)
        image[~mask] = min_value
        viz_map = cv2.applyColorMap(image, color)
    else:
        dpi = 100
        size = np.asarray(map_.shape) / dpi
        import matplotlib.pyplot as plt
        plt.subplots_adjust(left=0.0, bottom=0.0, top=1.0, right=1.0)
        fig = plt.figure(figsize=(size[1], size[0]), dpi=dpi)
        fig.clear()
        ax = fig.gca()
        ax.axis('off')
        ax.imshow(map_, cmap=color)
        canvas = fig.canvas
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.get_size_inches() * fig.get_dpi()
        viz_map = copy.deepcopy(image_flat.reshape(int(height), int(width), 3))
        plt.close()
    viz_map[~mask] = 0
    if bar_size is None:
        return viz_map
    bar = np.repeat(np.arange(0, bar_size[1])[None, :], bar_size[0], axis=0)
    bar = ((bar - bar.min()) * 255 / (bar.max() - bar.min())).astype(np.uint8)
    bar = cv2.applyColorMap(bar, color)
    return [viz_map, bar]


def apply_colormap(image: np.ndarray[np.uint8],
                   cmap: typing.Dict[int, typing.List[np.uint8]] = None,
                   bar_size: list[int] | np.ndarray[int] = None) -> np.ndarray | list[np.ndarray]:
    assert image.ndim == 2, 'image must be 8UC1, given channel: {}'.format(image.ndim)
    if cmap is None:
        cmap = {
            0: [128, 128, 128],  # ground
            1: [255, 255, 255],  # wire
            2: [255, 0, 0],  # obstacle
            3: [0, 0, 0],  # other
            4: [0, 0, 255],  # baseboard
        }
    categories = ['ground', 'wire', 'other']
    shape = list(image.shape) + [3]
    ret = np.zeros(shape, dtype=np.uint8)
    for cid, color in cmap.items():
        ret[image == cid] = color
    if bar_size is None:
        return ret
    bar = np.zeros(bar_size + [3], dtype=np.uint8)
    num_category = len(categories)
    width = bar_size[1] // num_category
    for c in range(num_category):
        start = c * width
        bar[:, start: (start + width)] = cmap[c]
        cv2.putText(bar, categories[c], (c * width, bar_size[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
    return [ret, bar]


class Padder:
    def __init__(self, height: int = None, width: int = None, size: list[int] = None, divide_by: int = 16) -> None:
        assert height is not None or size is not None, 'either height or size must be provided'
        assert width is not None or size is not None, 'either width or size must be provided'
        self._height = height if height is not None else size[0]
        self._width = width if width is not None else size[1]
        pad_ht = (((self._height // divide_by) + 1) * divide_by - self._height) % divide_by
        pad_wd = (((self._width // divide_by) + 1) * divide_by - self._width) % divide_by
        self._pad = None
        if any([pad_ht, pad_wd]):
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._pad is None:
            return x
        assert x.ndim >= 2, x.ndim
        return func.pad(x, self._pad, **kwargs)

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.pad(x, **kwargs)

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad is None:
            return x
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class ConvexUpsample(nn.Module):
    def __init__(self, inc: int, ouc: int, scale: int, mask: typing.Callable = None):
        """
          inc: input feature channel
          ouc: output feature channel
          scale: upsample scale. only support integer
        """
        super(ConvexUpsample, self).__init__()

        assert isinstance(scale, int), f'Scale must be integer, given {type(scale)}'

        self.mask = mask
        if mask is None:
            self.mask = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(inc),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inc, 9*scale*scale, kernel_size=3, stride=1, padding=1))
        weights= func.one_hot(torch.tensor([0,1,2,3,4,5,6,7,8]), num_classes=9).float().view(9,1,3,3)
        if ouc != 1:
            weights = weights.repeat(ouc, 1, 1, 1)
        self.weights = weights
        self.scale = scale

    def forward(self, x, feature = None):
        if feature is not None:
            mask = self.mask(feature)
        else:
            mask = self.mask(x)

        if not self.weights.is_cuda:
            self.weights = self.weights.to(x.device)

        x_unfold = func.conv2d(x, self.weights, padding=1, groups=x.shape[1])
        weight = func.pixel_shuffle(mask ,self.scale)
        b, c, h, w = weight.shape
        # weight: b, 9, 1, h*w
        weight = func.softmax(weight, dim=1).reshape(b, c, 1, h*w)
        # x: b, 9*inc, h, w => b, 9, inc, h*w
        x = func.interpolate(x_unfold, scale_factor=self.scale, mode='nearest')
        x = x.reshape(b, 9, -1, h*w)
        return (x*weight).sum(dim=1).reshape(b, -1, h, w)


def mask_by_disparity(disp: typing.Union[np.ndarray, torch.Tensor],
                      min_disparity: float,
                      max_disparity: float,
                      clip: bool = False) -> typing.Union[np.ndarray, torch.Tensor]:
    """
        this function will change the value of disp given clip = True
    """
    mask = np.full(disp.shape, 255, dtype=np.uint8)
    if min_disparity is not None:
        min_disp_mask = np.where(disp <= min_disparity, 0, 255).astype(np.uint8)
        mask = mask & min_disp_mask
    if max_disparity is not None:
        max_disp_mask = np.where(disp > max_disparity, 0, 255).astype(np.uint8)
        mask = mask & max_disp_mask
    if clip:
        np.clip(disp, min_disparity, max_disparity, out=disp)
    return mask

def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


"""
    camera matrix: 3x3
                    | fx   0  cx |
    camera matrix = |  0  fy  cy |
                    |  0   0   1 |
    distortion coefficients: 1x5
    distortion coefficients = [k1, k2, p1, p2, k3, k4]
"""
def load_params(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    config = configparser.ConfigParser()
    config.read(path)

    fx = config.getfloat('calibration', 'fx')
    fy = config.getfloat('calibration', 'fy')
    cx = config.getfloat('calibration', 'cx')
    cy = config.getfloat('calibration', 'cy')
    k1 = config.getfloat('calibration', 'k1')
    k2 = config.getfloat('calibration', 'k2')
    p1 = config.getfloat('calibration', 'p1')
    p2 = config.getfloat('calibration', 'p2')
    k3 = config.getfloat('calibration', 'k3')
    k4 = config.getfloat('calibration', 'k4')

    camera_matrix = build_camera_matrix(fx, fy, cx, cy)
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    fisheys_coeffs = np.array([k1, k2, k3, k4])

    width = config.getint('sensor', 'width')
    height = config.getint('sensor', 'height')

    return camera_matrix, dist_coeffs, fisheys_coeffs, width, height


def _undistort(array: np.ndarray,
               camera_matrix: np.ndarray,
               dist_coeffs: np.ndarray,
               fisheye_coeffs: np.ndarray,
               mode='pinhole',
               width: int=None,
               height: int=None,
               new_camera_matrix: np.ndarray = None) -> np.ndarray:
    if width is None or height is None:
        width, height = array.shape[1], array.shape[0]
    if mode == 'pinhole':
        if new_camera_matrix is None:
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), 5)
    elif mode == 'fisheye':
        if new_camera_matrix is None:
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, fisheye_coeffs, (width, height), np.eye(3), balance=1)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(camera_matrix, fisheye_coeffs, np.eye(3), new_camera_matrix, (width, height), cv2.CV_16SC2)

    dst = cv.remap(array, mapx, mapy, cv.INTER_LINEAR)
    #dst = cv.remap(array, mapx, mapy, cv.INTER_NEAREST)
    return dst
    # x, y, w, h = roi
    # return dst[y:y+h, x:x+w]

def undistort(array: np.ndarray,
              path: str = None,
              intrinsics: typing.Union[np.ndarray, typing.List[float]] = None,
              distortion: typing.Union[np.ndarray, typing.List[float]] = None,
              mode='pinhole',
              width: int=None,
              height: int=None,
              new_camera_matrix: np.ndarray = None) -> np.ndarray:
    if path is None:
        assert intrinsics is not None, 'camera matrix must be provided'
        assert len(intrinsics) == 4, 'camera matrix must be 4 elements [fx, fy, cx, cy]'
        assert distortion is not None, 'distortion coefficients must be provided'
        camera_matrix = build_camera_matrix(intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
        if mode == 'pinhole':
            assert len(distortion) >= 5, 'distortion coefficients must be 5 elements [k1, k2, p1, p2, k3]'
            dist_coeffs = np.array(distortion)
            fisheye_coeffs = None
        elif mode == 'fisheye':
            assert len(distortion) == 4, 'fisheye distortion coefficients must be 4 elements [k1, k2, k3, k4]'
            dist_coeffs = None
            fisheye_coeffs = np.array(distortion)
    else:
        camera_matrix, dist_coeffs, fisheye_coeffs, _, _ = load_params(path)
    return _undistort(array, camera_matrix, dist_coeffs, fisheye_coeffs, mode, width, height, new_camera_matrix)
