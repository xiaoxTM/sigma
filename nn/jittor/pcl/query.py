from .base import calculate_square_distance,calculate_square_distance_cn,calculate_square_distance_nc
import jittor as jt

def knn_query(num_samples, points, centroid_points, data_format='nc'):
    # [B,N,N]
    inverse_distance = -calculate_square_distance(centroid_points,points,data_format=data_format)
    _, idx = inverse_distance.topk(k=num_samples, dim=-1)  # (batch_size, num_points, num_samples)
    # [B,N,K]
    return idx


def ball_query_nc(num_samples,radius,points,centroid_points):
    B, N, _ = points.shape
    _, S, _ = centroid_points.size()
    idx = jt.arange(N, dtype='int64').view(1, 1, N).repeat([B, S, 1])
    sqrdists = calculate_square_distance_nc(centroid_points, points)
    idx[sqrdists > radius ** 2] = N
    idx = idx.argsort(dim=-1)[1][:, :, :num_samples]
    group_first = idx[:, :, 0].view(B, S, 1).repeat([1, 1, num_samples])
    mask = idx == N
    idx[mask] = group_first[mask]
    return idx


def ball_query_cn(num_samples,radius,points, centroid_points):
    '''
        :param: num_samples: int, number of samples to query
        :param: radius: float, radius of ball to query
        :param: points, Tensor, points to be searched (queried), [B,C,N]
    '''
    B, _, N = points.shape
    _, _, S = centroid_points.size()
    idx = jt.arange(N, dtype='int64').view(1, 1, N).repeat([B, S, 1])
    sqrdists = calculate_square_distance_cn(centroid_points, points)
    idx[sqrdists > radius ** 2] = N
    idx = idx.argsort(dim=-1)[1][:, :, :num_samples]
    group_first = idx[:, :, 0].view(B, S, 1).repeat([1, 1, num_samples])
    mask = idx == N
    idx[mask] = group_first[mask]
    return idx


def ball_query(num_samples,radius,points,centroid_points=None,data_format='nc'):
    if centroid_points is None:
        centroid_points = points
    if data_format == 'nc':
        return ball_query_nc(num_samples,radius,points, centroid_points)
    return ball_query_cn(num_samples,radius,points, centroid_points)


def ball(num_samples,radius,points,data_format='nc'):
    return ball_query(num_samples,radius,points,data_format=data_format)

# if __name__ == '__main__':
#     import numpy as np
#     import torch

#     t1 = torch.from_numpy(np.random.randn(2,10,3))
#     t2 = torch.from_numpy(np.random.randn(2,10,3))
#     i1 = torch.from_numpy(np.random.randint(0,2,size=(2,4)))

#     # gather_points
#     gp1 = gather_points_nc(t1,i1)
#     # print(gp1.size())
#     gp2 = gather_points_cn(t1.permute(0,2,1),i1)
#     # print(gp2.size())
#     print(gp1-gp2.permute(0,2,1))

#     # ball query
#     bq1 = ball_query_nc(4,0.01,t1,gp1)
#     bq2 = ball_query_cn(4,0.01,t1.permute(0,2,1),gp1.permute(0,2,1))
#     print(bq1)
#     print(bq2)    
#     print(bq1 - bq2.permute(0,2,1))