import os
import sys
sys.path.append('/home/xiaox/studio/src/git-series')

import pcl
from pcl import pointnet2_utils as std
from pcl import base, query, sampling
from pcl import interpolation as interp
from pcl.cpp.fps import farthest_point_sample


if __name__ == '__main__':
    import numpy as np
    import torch

    t1 = torch.from_numpy(np.random.randn(2,10,3).astype(np.float32))
    t2 = torch.from_numpy(np.random.randn(2,10,3).astype(np.float32))

    # print('test calculate_square_distance ---- ') # OK
    # d1 = base.calculate_square_distance_nc(t1,t2)
    # d2 = std.square_distance(t1,t2)
    # print(d1-d2)

    # print('test gather points ---- ') # OK
    # i1 = torch.from_numpy(np.random.randint(0,10,size=(2,4)))
    # g1 = query.gather_points_nc(t1,i1)
    # g2 = std.index_points(t1,i1)
    # print(g1-g2)

    print('test farthest point sample ---- ') # OK
    np.random.seed(1)
    torch.manual_seed(1)
    f1 = sampling.farthest_point_sample_nc(t1,5)
    np.random.seed(1)
    torch.manual_seed(1)
    f2 = std.farthest_point_sample(t1,5)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    f3 = farthest_point_sample(t1.to('cuda:0'),5).cpu()
    print(f1-f2)
    print(f1-f3)

    # print('test group points ---- ') # OK
    # i2 = torch.from_numpy(np.random.randint(0,10,size=(2,10,4)))
    # j1 = base.group_points_nc(t1,i2)
    # j2 = std.index_points(t1,i2)
    # print(j1-j2)

    # print('test ball query ---- ') # OK
    # i3 = torch.from_numpy(np.random.randint(0,10,size=(2,4)))
    # c1 = query.gather_points_nc(t1,i3)
    # print('c1 size:',c1.size())
    # b1 = query.ball_query_nc(3,0.01,t1,c1)
    # b2 = std.query_ball_point(0.01,3,t1,c1)
    # print(b1-b2)

    # print('test sample and group ---- ') # OK
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s1 = sample_and_group_nc(5,t1,3,0.01)[3]
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s2 = std.sample_and_group(5,0.01,3,t1,None,True)[2]
    # print(s1-s2)

    # print('test sample and group all ---- ') # OK
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s1 = pcl.sample_and_group_all_nc(t1,None)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s2 = std.sample_and_group_all(t1,None)[1]
    # print(s1-s2)