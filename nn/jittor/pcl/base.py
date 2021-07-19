import jittor as jt
import numpy as np


def calculate_square_distance_nc(t1, t2):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = t1.shape
    _, M, _ = t2.shape
    dist = -2 * jt.matmul(t1, t2.permute(0, 2, 1))
    dist += jt.sum(t1 ** 2, -1).view(B, N, 1)
    dist += jt.sum(t2 ** 2, -1).view(B, 1, M)
    return dist

def calculate_square_distance_cn(t1, t2):
    B, _, N = t1.size()
    B, _, M = t2.size()
    inner = -2 * jt.matmul(t1.transpose(2, 1), t2) # N x M
    xx = jt.sum(t1 ** 2, dim=1, keepdim=True).view(B,N,1)
    yy = jt.sum(t2 ** 2, dim=1, keepdim=True).view(B,1,M)
    distance = xx + inner + yy
    return distance

def calculate_square_distance(t1, t2=None, data_format='nc'):
    if t2 is None:
        t2 = t1
    if data_format == 'nc':
        return calculate_square_distance_nc(t1,t2)
    return calculate_square_distance_cn(t1,t2)

def group_points_nc(points, index):
    '''
       group points according to the index, i.e., group local points for each centroid point according to index
       :param: points: tensor, [B,N,C,...]
       :param: index: tensor, [B,S,K]
       :return: grouped points with shape of [B,S,K,C,...]
    '''
    batch_size,num_points = points.size()[:2]
    k = index.size(2)
    s = index.size(1)

    view_shape = [1] * len(index.size())
    view_shape[0] = batch_size

    idx_base = jt.arange(0, batch_size).view(*view_shape)*num_points

    index = index + idx_base
    index = index.view(-1)
    grouped_points = points.view(batch_size*num_points, -1)[index, :]
    channel_size = points.size()[2:]
    grouped_points = grouped_points.view(batch_size, s, k, *channel_size)
    return grouped_points

def group_points_cn(points, index):
    '''
       group points according to the index, i.e., group local points for each centroid point according to index
       :param: points: tensor, [B,C,...,N]
       :param: index: tensor, [B,S,K]
       :return: grouped points with shape of [B,C,...,,S,K]
    '''
    batch_size,num_points = points.size(0),points.size(-1)
    k = index.size(2)
    s = index.size(1)

    view_shape = [1] * len(index.size())
    view_shape[0] = batch_size

    idx_base = jt.arange(0, batch_size).view(view_shape)*num_points
    index = index + idx_base
    index = index.view(-1)
    # [B,C,...,N] -> [B,N,C,...]
    trans_size = [0, len(points.size())-1] + np.arange(1, len(points.size())-1).tolist()
    x = points.permute(*trans_size).contiguous()
    grouped_points = x.view(batch_size*num_points, -1)[index, :]
    # [B,N,-1] -> [B,N,K,C,...] -> [B,C,...,N,K]
    channel_size = points.size()[1:-1]
    grouped_points = grouped_points.view(batch_size, s, k, *channel_size)
    # print(grouped_points.size())
    output_dim_order = [0]+np.arange(3,len(grouped_points.size())).tolist()+[1,2]
    # print(output_dim_order)
    grouped_points = grouped_points.permute(*output_dim_order).contiguous()
    return grouped_points

def group_points(points, index, data_format='nc'):
    if data_format == 'nc':
        return group_points_nc(points, index)
    return group_points_cn(points, index)


def gather_points_nc(points, index):
    """
    gathering points according to the index, i.e., gathering the subset of the points (downsampling)
    Input:
        points: input points data, [B, N, C, ...]
        index: sample index data, [B, S]
    Return:
        gathered_points:, indexed points data, [B, S, C, ...]
    """
    B = points.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = jt.arange(B, dtype='int64').view(view_shape).repeat(repeat_shape)
    gathered_points = points[batch_indices, index, :]
    return gathered_points


def gather_points_cn(points, index):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    B = points.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = jt.arange(B).view(view_shape).repeat(repeat_shape)
    gathered_points = points[batch_indices, :, index]
    return gathered_points.permute(0,2,1)


def gather_points(points, index, data_format='nc'):
    if data_format == 'nc':
        return gather_points_nc(points, index)
    return gather_points_cn(points,index)


def knn(x, k, data_format='nc'):
    inverse_distance = -calculate_square_distance(x,data_format=data_format)
    _, idx = inverse_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    # [B,N,K]
    return idx, inverse_distance


# if __name__ == '__main__':
#     import numpy as np
#     t1 = torch.from_numpy(np.random.randn(2,10,3,4,5))
#     t2 = torch.from_numpy(np.random.randn(2,10,3,4,5))
#     i1 = torch.from_numpy(np.random.randint(0,10,size=(2,8,4)))

#     d1 = group_points_nc(t1,i1) # [2,8,k,3,4,5]
#     d2 = group_points_cn(t1.permute(0,2,3,4,1),i1) # [2,3,4,5,8,k]
#     print(d1-d2.permute(0,4,5,1,2,3))