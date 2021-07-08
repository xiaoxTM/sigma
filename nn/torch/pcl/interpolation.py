from .base import calculate_square_distance_nc, calculate_square_distance_cn, group_points_cn, group_points_nc


def interpolate_cn(sampled_features, points, sampled_points, num_neighbors=3):
    """
    Input:
        points: input points position data, [B, C, N]
        sampled: sampled input points position data, [B, C, S]
        sampled_features: input points data, [B, D, S]
    Return:
        upsamled_points: upsampled points data, [B, D, N]
    """

    B, C, N = points.shape
    _, _, S = sampled_points.shape

    if S == 1:
        upsampled_points = sampled_features.repeat(1, 1, N)
    else:
        dists = calculate_square_distance_cn(points, sampled_points) # [B, N, S]
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :num_neighbors], idx[:, :, :num_neighbors]  # [B, N, num_neighbors]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # [B,C,S]
        gathered_points = group_points_cn(sampled_features, idx)
        # print(gathered_points.size())
        upsampled_features = torch.sum(gathered_points * weight.view(B, 1, N, num_neighbors), dim=3)
    return upsampled_features, idx


def interpolate_nc(sampled_features, points, sampled_points, num_neighbors=3):
    """
    Input:
        points: input points position data, [B, N, C]
        sampled: sampled input points position data, [B, S, C]
        sampled_features: input points data, [B, S, D]
    Return:
        upsamled_points: upsampled points data, [B, N, D]
    """

    B, N, C = points.shape
    _, S, _ = sampled_points.shape

    if S == 1:
        upsampled_points = sampled_features.repeat(1, N, 1)
    else:
        dists = calculate_square_distance_nc(points, sampled_points)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :num_neighbors], idx[:, :, :num_neighbors]  # [B, N, num_neighbors]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        #                                 [B,S,C]
        # print('samples features size:', sampled_features.size(),'index size:',idx.size())
        gathered_points = group_points_nc(sampled_features, idx)
        # print(gathered_points.size())
        upsampled_features = torch.sum(gathered_points * weight.view(B, N, num_neighbors, 1), dim=2)
    return upsampled_features, idx

def interpolate(sampled_features, points, sampled_points, num_neighbors, data_format='nc'):
    if data_format == 'nc':
        return interpolate_nc(sampled_features,points,sampled_points,num_neighbors)
    return interpolate_cn(sampled_features,points,sampled_points,num_neighbors)


# if __name__ == '__main__':
#     import numpy as np
#     import torch

#     np.random.seed(1)
#     torch.manual_seed(1)

#     t1 = torch.from_numpy(np.random.randn(2,10,3).astype(np.float32))
#     t2 = torch.from_numpy(np.random.randn(2,4,3).astype(np.float32))
#     f1 = torch.from_numpy(np.random.randn(2,4,5))

#     i1 = interpolate_nc(f1,t1,t2,4)[0]
#     # print(i1.size())
#     i2 = interpolate_cn(f1.permute(0,2,1),t1.permute(0,2,1),t2.permute(0,2,1),4)[0]
#     # print(i2.size())
#     print(i1-i2.permute(0,2,1))
#     # print(i1-i2)