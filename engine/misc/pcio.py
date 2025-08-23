import numpy as np
from . import transformation as trans


def remove_invalid_points(points: np.ndarray) -> np.ndarray:
    mask = np.any(np.logical_or(np.isnan(points), np.isinf(points)), axis=-1)
    return points[~mask]


def save_pointcloud_with_color(filename: str, points: np.ndarray):
    formatted_points = []
    pointclouds = remove_invalid_points(points.reshape((-1, 6)))
    for point in pointclouds:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(filename, "w")
    out_file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
    ''' % (len(pointclouds), "".join(formatted_points)))
    out_file.close()

def save_pointcloud_with_intensity(filename: str, points: np.ndarray):
    formatted_points = []
    pointclouds = remove_invalid_points(points.reshape((-1, 4)))
    for point in pointclouds:
        formatted_points.append("%f %f %f %f\n" % (point[0], point[1], point[2], point[3]))

    out_file = open(filename, "w")
    out_file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property float intensity
end_header
%s
    ''' % (len(pointclouds), "".join(formatted_points)))
    out_file.close()


def save_pointcloud(filename: str, points: np.ndarray):
    assert points.ndim == 2 or points.ndim == 3, "Invalid pointcloud shape dim: {}".format(points.ndim)
    if points.ndim == 3:
        channel = points.shape[-1]
        points = points.reshape(-1, channel)
    if points.shape[1] == 4:
        save_pointcloud_with_intensity(filename, points)
    elif points.shape[1] == 6:
        save_pointcloud_with_color(filename, points)
    else:
        pointclouds = remove_invalid_points(points)
        formatted_points = []
        for point in pointclouds:
            formatted_points.append("%f %f %f\n" % (point[0], point[1], point[2]))

        out_file = open(filename, "w")
        out_file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s
        ''' % (len(pointclouds), "".join(formatted_points)))
        out_file.close()


def load_pointcloud(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.read().split('end_header')[1].splitlines()
    pointcloud = []
    length = None
    for idx, line in enumerate(lines):
        buffer = np.fromstring(line, sep=' ')
        if length is None:
            if buffer.shape[0] > 0:
                length = buffer.shape[0]
                pointcloud.append(buffer)
            else:
                print('warning: unexpected 0 element size')
        elif buffer.shape[0] == length:
            pointcloud.append(buffer)
        else:
            print('warning: unexpected element size, given {} expected: {}'.format(buffer.shape[0], length))

    return np.asarray(pointcloud)


def save_depth_as_pointcloud(filename: str, depth: np.ndarray, image: np.ndarray = None, **kwargs):
    pc = trans.depth_to_pointcloud(depth, **kwargs)
    if image is not None:
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        pc = np.concatenate([pc, image], axis=2)
    save_pointcloud(filename, pc)
