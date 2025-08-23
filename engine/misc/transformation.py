import numpy as np
import typing
# import open3d as o3d

# from src.misc import pcio


def disparity_to_depth(disparity: np.ndarray, baseline: float, focus: float) -> np.ndarray:
    depth = np.zeros_like(disparity)
    nonzero_mask = disparity > 0
    # assert nonzero_mask.sum() == depth.size, 'all depth values are zero ({} vs {})'.format(nonzero_mask.sum(), depth.size)
    depth[nonzero_mask] = focus * baseline / disparity[nonzero_mask]
    return depth


def depth_to_disparity(depth: np.ndarray, baseline: float, focus: float) -> np.ndarray:
    disparity = np.zeros_like(depth)
    nonzero_mask = depth > 0
    # assert nonzero_mask.sum() == depth.size, 'all depth values are zero ({} vs {})'.format(nonzero_mask.sum(), depth.size)
    disparity[nonzero_mask] = focus * baseline / depth[nonzero_mask]
    return disparity


def depth_to_pointcloud(depth: np.ndarray,
                        fx: float = None,
                        cx: float = None,
                        fy: float = None,
                        cy: float = None,
                        intrinsics: typing.Union[np.ndarray, typing.List[float]] = None) -> np.ndarray:
    if fx is None:
        if fy is None:
            fy = intrinsics[1]
        if intrinsics[0] is None:
            raise ValueError('both `fx` and intrinsics[0, 0] are None. you must at least specify one of them')
        fx = intrinsics[0]
    if fy is None:
        fy = fx

    if cx is None:
        if cy is None:
            cy = intrinsics[3]
        if intrinsics[2] is None:
            raise ValueError('both `cx` and intrinsics[0, 2] are None. you must at least specify one of them')
        cx = intrinsics[2]
    if cy is None:
        cy = cx

    gridx, gridy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))

    extra_info = None
    if depth.ndim == 3:
        # h, w, c
        extra_info = depth[..., 1:]
        depth = depth[..., 0]

    worldz = depth
    worldx = worldz * (gridx.astype(np.float32) - cx) / fx
    worldy = worldz * (gridy.astype(np.float32) - cy) / fy

    ret = np.stack([worldx, worldy, worldz], axis=2)
    if extra_info is not None:
        if extra_info.ndim == 2:
            extra_info = np.expand_dims(extra_info, axis=2)
        ret = np.concatenate([ret, extra_info], axis=2)
    return ret


def pointcloud_to_depth(pointcloud: np.ndarray,
                        fx: float = None,
                        cx: float = None,
                        fy: float = None,
                        cy: float = None,
                        intrinsics: typing.Union[np.ndarray, typing.List[float]] = None,
                        extrinsics: typing.Union[np.ndarray, typing.List[float]] = None,
                        map_size: tuple[int, int] = None) -> np.ndarray:
    if map_size is None:
        return pointcloud[..., 2]

    if fx is None:
        if fy is None:
            fy = intrinsics[1]
        if intrinsics[0] is None:
            raise ValueError('both `fx` and intrinsics[0, 0] are None. you must at least specify one of them')
        fx = intrinsics[0]
    if fy is None:
        fy = fx

    if cx is None:
        if cy is None:
            cy = intrinsics[3]
        if intrinsics[2] is None:
            raise ValueError('both `cx` and intrinsics[0, 2] are None. you must at least specify one of them')
        cx = intrinsics[2]
    if cy is None:
        cy = cx

    # 3 x 4
    intrinsics = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float32)

    # 4 x 4
    if extrinsics is None:
        extrinsics = np.eye(4).astype(np.float32)

    if pointcloud.ndim > 2:
        # [H, W, C] -> [-1, C]
        C = pointcloud.shape[-1]
        pointcloud = pointcloud.reshape(-1, C)

    extra_info = None
    if pointcloud.shape[-1] > 3:
        extra_info = pointcloud[:, 3:]
        pointcloud = pointcloud[:, :3]

    # [N, 3] -> [N, 4]
    pointcloud = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1), dtype=np.float32)], axis=1)

    # | uz |   | fx 0  cx 0 |   | R  t |   | X |
    # | vz | = | 0  fy cy 0 | x |      | x | Y |
    # | z  |   | 0  0  1  0 |   | O  1 |   | Z |
    #                                      | 1 |
    # R: 3 x 3
    # t: 3 x 1
    # O: 1 x 3
    # z_uv: 3 x N
    z_uv = intrinsics @ extrinsics @ pointcloud.T

    depth = np.zeros(map_size, dtype=np.float32)
    U = np.rint(z_uv[0, :] / z_uv[2, :]).astype(np.int32)
    V = np.rint(z_uv[1, :] / z_uv[2, :]).astype(np.int32)
    mask = (U >= 0) & (U < map_size[1]) & (V >= 0) & (V < map_size[0]) & (z_uv[2, :] > 0) & (z_uv[2, :] < 16)
    depth[V[mask], U[mask]] = z_uv[2, mask]  # [H, W]
    if extra_info is None:
        return depth
    extra = np.zeros((map_size[0], map_size[1], extra_info.shape[-1]), dtype=extra_info.dtype)
    extra[V[mask], U[mask]] = extra_info[mask, :]
    return np.concatenate([np.expand_dims(depth, axis=2), extra], axis=2)


def distance_to_peak(distance: np.ndarray, bin_size: float = 416e-12) -> np.ndarray:
    light_speed = 2.99792458e8
    factor = bin_size * light_speed
    peak = distance * 2 / factor
    return peak


def peak_to_distance(peak: np.ndarray, bin_size: float = 416e-12) -> np.ndarray:
    light_speed = 2.99792458e8
    return peak * bin_size * light_speed / 2


def depth_to_distance(depth: np.ndarray, fx: float, cx: float, fy: float= None, cy: float = None) -> np.ndarray:
    gridx, gridy =np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    if fy is None:
        fy = fx
    if cy is None:
        cy = cx
    distance = depth * np.sqrt(((gridx-cx) / fx)**2 + ((gridy-cy) / fy)**2 + 1)
    return distance


def distance_to_depth(distance: np.ndarray, fx: float, cx: float, fy: float = None, cy: float = None) -> np.ndarray:
    gridx, gridy =np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]), indexing='xy')
    if fy is None:
        fy = fx
    if cy is None:
        cy = cx
    depth = distance / np.sqrt(((gridx-cx) / fx)**2 + ((gridy-cy) / fy)**2 + 1)
    return depth


def peak_to_depth(peak: np.ndarray, fx: float, cx: float, fy: float = None, cy: float = None, bin_size: float = 416e-12) -> np.ndarray:
    distance = peak_to_distance(peak, bin_size)
    return distance_to_depth(distance, fx, cx, fy, cy)


def depth_to_peak(depth: np.ndarray, fx: float, cx: float, fy: float, cy: float = None, bin_size: float = 416e-12) -> np.ndarray:  # checked OK
    distance = depth_to_distance(depth, fx, cx, fy, cy)
    ret = distance_to_peak(distance, bin_size)
    # print('depth[', 25, 118, ']=', depth[25, 118], '-->', 'distance[', 25, 118, ']=', distance[25, 118], '-->', ret[25, 118])
    return ret


def peak_to_pointcloud(peak: np.ndarray,
                       fx: float,
                       cx: float,
                       fy: float = None,
                       cy: float = None,
                       bin_size: float = 416e-12) -> np.ndarray:
    depth = peak_to_depth(peak, fx, cx, fy, cy, bin_size)
    return depth_to_pointcloud(depth, fx, cx, fy, cy)


def pointcloud_to_peak(pointcloud: np.ndarray,
                       fx: float,
                       cx: float,
                       fy: float = None,
                       cy: float = None,
                       bin_size: float = 416e-12) -> np.ndarray:
    depth = pointcloud_to_depth(pointcloud, fx, cx, fy, cy)
    return depth_to_peak(depth, fx, cx, fy, cy, bin_size)


def disparity_to_pointcloud(disparity: np.ndarray,
                            baseline: float,
                            fx: float = None,
                            cx: float = None,
                            fy: float = None,
                            cy: float = None,
                            intrinsics: typing.Union[np.ndarray, typing.List[float]] = None) -> np.ndarray:
    depth = disparity_to_depth(disparity, baseline, fx if fx is not None else intrinsics[0])
    return depth_to_pointcloud(depth, fx, cx, fy, cy, intrinsics)


def disparity_to_peak(disparity: np.ndarray,
                      baseline: float,
                      fx: float = None,
                      cx: float = None,
                      fy: float = None,
                      cy: float = None,
                      bin_size: float = 416e-12) -> np.ndarray:
    depth = disparity_to_depth(disparity, baseline, fx)
    # depth[depth > 15950] = 0 # out of dToF range, 15m = 15000mm
    return depth_to_peak(depth, fx, cx, fy, cy, bin_size)
