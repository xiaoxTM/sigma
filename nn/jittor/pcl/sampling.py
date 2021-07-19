import jittor as jt
from .base import gather_points_nc,group_points_nc
from .query import knn_query, ball_query_nc

def farthest_point_sample_nc(points, num_samples):
    """
    Input:
        points: pointcloud data, [B, N, 3]
        num_samples: number of samples
    Return:
        centroids_index: sampled pointcloud index, [B, num_samples]
    """
    B, N, _ = points.shape
    centroids_index = jt.zeros((B, num_samples),dtype='int64')
    distance = jt.ones((B, N)) * 1e10
    # farthest = jt.ones((B,),dtype='int64')
    farthest = jt.randint(0, N, (B,), dtype='int64')
    batch_indices = jt.arange(B, dtype='int64')
    for i in range(num_samples):
        # print('farthest:',farthest.size())
        centroids_index[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = jt.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = jt.argmax(distance, -1)[0]
    return centroids_index

def farthest_point_sample_cn(points, num_samples):
    B, _, N = points.shape
    centroids_index = jt.zeros((B, num_samples), dtype='int64')
    distance = jt.ones((B, N)) * 1e10
    farthest = jt.randint(0, N, (B,), dtype='int64')
    batch_indices = jt.arange(B, dtype='int64')
    for i in range(num_samples):
        centroids_index[:, i] = farthest
        centroid = points[batch_indices, :, farthest].view(B,3,1)
        dist = jt.sum((points - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = jt.argmax(distance, -1)[0]
    return centroids_index

def farthest_point_sample(points, num_samples, data_format='nc'):
    if data_format == 'nc':
        return farthest_point_sample_nc(points, num_samples)
    return farthest_point_sample_cn(points, num_samples)

def sample_and_group_nc(num_points, points, num_samples, radius=None, features=None):
    """
    Input:
        npoint:
        radius:
        nsample:
        points: input points position data, [B, N, 3]
        features: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        grouped_features: sampled points data, [B, npoint, nsample, 3+D]
    """
    # B, N, C = points.shape
    centroid_idx = farthest_point_sample_nc(points, num_points) # [B, num_point]
    centroid_points = gather_points_nc(points, centroid_idx)
    if radius is None:
        idx = knn_query(num_samples, points, centroid_points, data_format='nc') # [B, num_points, num_samples]
    else:
        idx = ball_query_nc(num_samples, radius, points, centroid_points)
    grouped_points = group_points_nc(points, idx) # [B, npoint, nsample, C]

    if features is not None:
        centroid_features = gather_points_nc(features, centroid_idx)
        grouped_features = group_points_nc(features, idx)
        return centroid_idx, idx, centroid_points, grouped_points, centroid_features, grouped_features
    return centroid_idx, idx, centroid_points, grouped_points

def sample_and_group_all_nc(points, features):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = points.shape
    # centroid_points = jt.zeros((B, 1, C))
    grouped_points = points.view(B, 1, N, C)
    if features is not None:
        grouped_features = features.view(B, 1, N, -1)
        return grouped_points, grouped_features 
    return grouped_points

# if __name__ == '__main__':
#     import numpy as np
#     import torch

#     np.random.seed(1)
#     torch.manual_seed(1)

#     t1 = torch.from_numpy(np.random.randn(2,10,3).astype(np.float32))
#     t2 = torch.from_numpy(np.random.randn(2,10,3))
#     i1 = torch.from_numpy(np.random.randint(0,2,size=(2,4)))

#     fps1 = farthest_point_sample_nc(t1, 5)
#     fps2 = farthest_point_sample_cn(t1.permute(0,2,1),5)
#     print(fps1-fps2)