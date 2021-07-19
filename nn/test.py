import os
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
# sys.path.append(os.getcwd())
# from torch.pcl import pointnet2_utils as std
import sigma.nn.pytorch.pcl as tpcl #import base, query, sampling
# from .torch.pcl import interpolation as interp
import sigma.nn.pyjittor.pcl as jpcl


if __name__ == '__main__':
    import numpy as np
    import torch
    import jittor as jt
    n1 = np.random.randn(2,10,3).astype(np.float32)
    n2 = np.random.randn(2,10,3).astype(np.float32)

    t1 = torch.from_numpy(n1)
    t2 = torch.from_numpy(n2)

    j1 = jt.array(n1)
    j2 = jt.array(n2)

    # print('test calculate_square_distance ---- ') # OK
    # d1 = tpcl.calculate_square_distance(t1,t2)
    # d2 = jpcl.calculate_square_distance(j1,j2)
    # print(d1.numpy()-d2.numpy())

    # print('test gather points ---- ') # OK
    # n3 = np.random.randint(0,10,size=(2,4))
    # ji1 = jt.array(n3)
    # ti1 = torch.from_numpy(n3)
    # g1 = tpcl.gather_points(t1,ti1)
    # g2 = jpcl.gather_points(j1,ji1)
    # print(g1.numpy()-g2.numpy())

    # print('test farthest point sample ---- ') # OK
    # np.random.seed(1)
    # torch.manual_seed(1)
    # f1 = tpcl.farthest_point_sample(t1,5,data_format='nc')
    # np.random.seed(1)
    # jt.set_seed(1)
    # f2 = jpcl.farthest_point_sample(j1, 5, data_format='nc')
    # print(f1.numpy()-f2.numpy())

    # print('test group points ---- ') # OK
    # n4 = np.random.randint(0,10,size=(2,5,3))
    # ti2 = torch.from_numpy(n4)
    # ji2 = jt.array(n4)
    # u1 = tpcl.group_points(t1,ti2)
    # u2 = jpcl.group_points(j1,ji2)
    # print(u1.numpy()-u2.numpy())

    # print('test ball query ---- ') # OK
    # i3 = torch.from_numpy(np.random.randint(0,10,size=(2,4)))
    # c1 = tpcl.gather_points(t1,i3)
    # b1 = tpcl.ball_query(3,0.01,t1,c1)
    # b2 = jpcl.ball_query(3,0.01,j1,jt.array(c1.numpy()))
    # print(b1.numpy()-b2.numpy())

    # print('test sample and group ---- ') # OK
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s1 = tpcl.sample_and_group_nc(5,t1,3,0.01)[3]
    # np.random.seed(1)
    # jt.set_global_seed(1)
    # s2 = jpcl.sample_and_group_nc(5,j1,3,0.01)[3]
    # print(s1.numpy()-s2.numpy())

    # print('test sample and group all ---- ') # OK
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s1 = tpcl.sample_and_group_all_nc(t1,None)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # s2 = jpcl.sample_and_group_all_nc(j1,None)
    # print(s1.numpy()-s2.numpy())