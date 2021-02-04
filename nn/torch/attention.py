import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sigma.nn.torch import initializers,activations,normalizations,utils


def attention(Q, K, V, scale=1.0, data_format='NCH'):
    if data_format.upper() == 'NHC':
        weights = torch.softmax(Q.bmm(K.transpose(2,1))*scale, 2)
        return weights.bmm(V), weights
    elif data_format.upper() == 'NCH':
        weights = torch.softmax(K.transpose(2,1).bmm(Q)*scale, dim=1)
        return V.bmm(weights), weights
    else:
        raise ValueError('`{}` data format not support'.format(data_format))

class Attention(nn.Module):
    ''' Attention
    '''
    def __init__(self,
                 num_dims=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 bias=True,
                 rank=3,
                 data_format='NCH'):
        super(Attention, self).__init__()
        assert data_format.upper() in ['NCH', 'NHC']
        self._data_format = data_format
        self._weight_initializer = initializers.get(weight_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

        self._scale = 1.0
        if num_dims is not None:
            self._scale = 1.0 / math.sqrt(num_dims)
        if data_format.upper() == 'NCH':
            self._attention = self._attention_nch
        else:
            self._attention = self._attention_nhc

    def _initialize(self, weight_parameters, bias_parameters):
        if weight_parameters is not None and self._weight_initializer is not None:
            if not isinstance(weight_parameters, (list, tuple)):
                weight_parameters = [weight_parameters]
            for wp in weight_parameters:
                self._weight_initializer(wp)
        if bias_parameters is not None and self._bias_initializer is not None:
            if not isinstance(bias_parameters, (list, tuple)):
                bias_parameters = [bias_parameters]
            for bp in bias_parameters:
                self._bias_initializer(bp)

    def _attention_nch(self, q, k, v):
        ''' q: batch-size, dim, num-points
            k: batch-size, dim, num-points
        '''
        # batch-size, dim, num-points
        weights = torch.softmax(k.transpose(2,1).bmm(q)*self._scale, dim=1)
        return v.bmm(weights), weights

    def _attention_nhc(self, q, k, v):
        ''' q: batch-size, num-points, dim
            k: batch-size, num-points, dim
        '''
        weights = torch.softmax(q.bmm(k.transpose(2,1))*self._scale, dim=2)
        return weights.bmm(v), weights

    def forward(self, Q, K, V):
        return self._attention(Q, K, V) # according to ``Attention Is All You Need''


class SelfAttention(Attention):
    ''' Attention
    '''
    def __init__(self,
                 cin,
                 cout=None,
                 scale=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 bias=True,
                 rank=3,
                 residual=False,
                 data_format='NCH'):
        if cout is None:
            cout = cin
        self.residual = residual
        super(SelfAttention, self).__init__(cout,weight_initializer,bias_initializer,bias,rank,data_format)
        if data_format == 'NHC':
            self._project_q = nn.Linear(cin, cout, bias)
            self._project_k = nn.Linear(cin, cout, bias)
            self._project_v = nn.Linear(cin, cout, bias)
        else:
            if rank == 3:
                conv = nn.Conv1d
            elif rank == 4:
                conv = nn.Conv2d
            elif rank == 5:
                conv = nn.Conv3d
            else:
                raise ValueError('data format[`NCH`] with rank[{}] not support'.format(rank))
            self._project_q = conv(cin, cout, kernel_size=1, bias=bias)
            self._project_k = conv(cin, cout, kernel_size=1, bias=bias)
            self._project_v = conv(cin, cout, kernel_size=1, bias=bias)
        self._initialize([self._project_q.weight.data,self._project_k.weight.data,self._project_v.weight.data],
                          [self._project_q.bias.data,self._project_k.bias.data,self._project_v.bias.data])

    def forward(self, x):
        q = self._project_q(x)
        k = self._project_k(x)
        v = self._project_v(x)
        y = self._attention(q, k, v) # according to ``Attention Is All You Need''
        if self.residual:
            y = y + v
        return y


class MultiheadAttention(nn.Module):
    ''' Multihead Attention
    '''
    def __init__(self,num_dim_q,
                 num_heads,
                 num_dim_v=None,
                 num_dim_mq=None,
                 num_dim_mv=None,
                 num_dim_o=None,
                 scale=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 bias=True,
                 rank=3,
                 data_format='NCH'):
        super(MultiheadAttention, self).__init__()
        assert data_format in ['NCH', 'NHC']
        self._data_format = data_format
        #if data_format.upper() != 'NHC':
        #    raise ValueError('MultiheadAttention does not support `{}` data format'.format(data_format))
        if num_dim_v is None:
            num_dim_v = num_dim_q
        if num_dim_mq is None:
            num_dim_mq = num_dim_q // num_heads
        if num_dim_mv is None:
            num_dim_mv = num_dim_v // num_heads
        if num_dim_o  is None:
            num_dim_o = num_dim_v
        assert num_dim_q > 0
        assert num_dim_v > 0
        assert num_dim_o > 0
        assert num_dim_mq > 0
        assert num_dim_mv > 0
        self._num_heads = num_heads
        self._num_dim_q = num_dim_q
        self._num_dim_v = num_dim_v
        self._num_dim_o = num_dim_o
        self._num_dim_mq = num_dim_mq
        self._num_dim_mv = num_dim_mv
        weight_initializer = initializers.get(weight_initializer)
        bias_initializer = initializers.get(bias_initializer)
        if data_format == 'NHC':
            self._project_q = nn.Linear(num_dim_q, num_dim_mq*num_heads, bias)
            self._project_k = nn.Linear(num_dim_q, num_dim_mq*num_heads, bias)
            self._project_v = nn.Linear(num_dim_v, num_dim_mv*num_heads, bias)
            self._project_o = nn.Linear(num_dim_mv*num_heads, num_dim_o, bias)
            self._multi_attention = self._multi_attention_nhc
        else:
            if rank == 3:
                conv = nn.Conv1d
            elif rank == 4:
                conv = nn.Conv2d
            elif rank == 5:
                conv = nn.Conv3d
            else:
                raise ValueError('data format[`NCH`] with rank[{}] not support'.format(rank))
            self._project_q = conv(num_dim_q, num_dim_mq*num_heads, kernel_size=1, bias=bias)
            self._project_k = conv(num_dim_q, num_dim_mq*num_heads, kernel_size=1, bias=bias)
            self._project_v = conv(num_dim_v, num_dim_mv*num_heads, kernel_size=1, bias=bias)
            self._project_o = conv(num_dim_mv*num_heads, num_dim_o, kernel_size=1, bias=bias)
            self._multi_attention = self._multi_attention_nch
        if weight_initializer is not None:
            weight_initializer(self._project_q.weight.data)
            weight_initializer(self._project_k.weight.data)
            weight_initializer(self._project_v.weight.data)
            weight_initializer(self._project_o.weight.data)
        if bias_initializer is not None and bias:
            bias_initializer(self._project_q.bias.data)
            bias_initializer(self._project_k.bias.data)
            bias_initializer(self._project_v.bias.data)
            bias_initializer(self._project_o.bias.data)
        self._scale = 1
        if scale:
            self._scale = 1.0 / math.sqrt(num_dim_q) # according to ``Set Transformer''

    def _multi_attention_nch(self,q,k,v):
        ''' q, k, v: shape of [batch-size, num-dim-x, num-points-x, ...]
        '''
        shape = list(q.size())
        qshape = [shape[0], self._num_dim_mq, self._num_heads] + [shape[2], 1] + shape[3:]
        # q: [batch-size, num-dim-mq, num-heads, num-points-q, 1, ...]
        q = q.view(*qshape)
        kshape = [shape[0], self._num_dim_mq, self._num_heads, 1, k.size(2)] + shape[3:]
        # k: [batch-size, num-dim-mq, num-heads, 1, num-points-v, ...]
        k = k.view(*kshape)
        # sum: [batch-size, 1, num-heads, num-points-q, num-points-v, ...]
        x = torch.softmax((q * k).sum(1, keepdim=True)*self._scale, dim=4)
        vshape = [shape[0], self._num_dim_mv, self._num_heads, 1, v.size(2)] + shape[3:]
        # v: [batch-size, num-dim-mv, num-heads, 1, num-points-v, ...]
        v = v.view(*vshape)
        # x: [batch-size, num-dim-mv, num-heads, num-points-q, ...]
        # x * v: [batch-size, num-dim-mv, num-heads, num-points-q, num-points-v, ...]
        # o: [batch-size, num-dim-mv, num-heads, num-points-q, ...]
        o = (x * v).sum(4)
        oshape = list(o.size())
        oshape = [oshape[0]] + [-1] + oshape[3:]
        return o.view(oshape)

    def _multi_attention_nhc(self,q,k,v):
        ''' q, k, v: shape of [batch-size, num-points, ..., num-dim-x]
        '''
        shape = list(q.size())
        qshape = shape[:2] + [1] + shape[2:-1] + [self._num_heads, self._num_dim_mq]
        # q: [batch-size, num-points-q, 1, ..., num-heads, num-dim-mq]
        q = q.view(qshape)
        kshape = [shape[0], 1, k.size(1)] + shape[2:-1] + [self._num_heads, self._num_dim_mq]
        # k: [batch-size, 1, num-points-v, ..., num-heads, num-dim-mq]
        k = k.view(*kshape)
        # sum: [batch-size, num-points-q, num-points-v, ..., num-heads, 1]
        x = torch.softmax((q * k).sum(-1, keepdim=True)*self._scale, dim=2)
        vshape = [shape[0], 1, v.size(1)] + shape[2:-1] + [self._num_heads, self._num_dim_mv]
        # v: [batch-size, 1, num-points-v, ..., num-heads, num-dim-mv]
        v = v.view(vshape)
        # x: [batch-size, num-points-q, ..., num-heads, num-dim-mv]
        # x * v: [batch-size, num-points-q, num-points-v, ..., num-heads, num-dim-mv]
        # o: [batch-size, num-points-q, ..., num-heads, num-dim-mv]
        o = (x * v).sum(dim=2)
        oshape = list(o.size())[:-2] + [-1]
        return o.view(oshape)

    def forward(self, Q, K=None, V=None):
        #  [batch-size, num-dim-x, num-heads, num-points] for data_format = 'NCH'
        #  [batch-size, num-points, num-heads, num-dim-x] for data_format = 'NHC'
        q = self._project_q(Q) # num_dim_q => num_heads, num_dim_mq
        k = self._project_k(K if K is not None else Q)
        v = self._project_v(V if V is not None else Q) # num_dim_v => num_head, num_dim_mv
        # [batch-size, num-dim-v, num-points-q, ...] for data_format = 'NCH'
        # [batch-size, num-points-q, ..., num-dim-v] for data_format = 'NHC'
        o = self._multi_attention(q, k, v)
        return self._project_o(o)


class MultiheadAttentionBlock(MultiheadAttention):
    ''' multihead attention block
    '''
    def __init__(self, num_dims, num_heads, norm='ln', act='relu<inplace:b:True>', **kwargs):
        # data_format == 'NHC'
        super(MultiheadAttentionBlock, self).__init__(num_dims,num_heads,**kwargs)
        assert self._num_dim_q == self._num_dim_v,\
               'dimension of Q and V must be the same. given {} vs {}'.format(self._num_dim_q,
                                                                              self._num_dim_v)
        assert self._num_dim_q == self._num_dim_o,\
               'dimension of Q and O must be the same. given {} vs {}'.format(self._num_dim_q,
                                                                              self._num_dim_o)
        self._norm0 = normalizations.get(norm,self._num_dim_o,'1d',size=[self._num_dim_v])
        self._norm1 = normalizations.get(norm,self._num_dim_o,'1d',size=[self._num_dim_v])

        self._act = activations.get(act)

    def forward(self, Q, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = K
        o = super(MultiheadAttentionBlock, self).forward(Q, K, V)
        if self._norm0 is not None:
            if self._data_format == 'NHC':
                o = self._norm0(Q + o)
            else:
                o = self._norm0((Q+o).transpose(1,-1)).transpose(1,-1)
        o = o + self._act(o)
        if self._norm1 is not None:
            if self._data_format == 'NHC':
                o = self._norm1(o)
            else:
                o = self._norm1(o.transpose(1,-1)).transpose(1,-1)
        return o

MAB = MultiheadAttentionBlock


class SetAttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SetAttentionBlock, self).__init__()
        self._mab = MAB(*args, **kwargs)

    def forward(self, x):
        return self._mab(x, x)

SAB = SetAttentionBlock


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, num_reduced_points, num_dims, initializer='xavier-uniform', *args, **kwargs):
        super(InducedSetAttentionBlock, self).__init__()
        self._i = nn.Parameter(torch.Tensor(num_reduced_points, num_dims))
        initializers.get(initializer)(self._i)
        self._mab0 = MAB(num_dims, *args, **kwargs)
        self._mab1 = MAB(num_dims, *args, **kwargs)

    def forward(self, x):
        ishape = [1] + list(self._i.size())
        i = self._i.view(ishape)
        h = self._mab0(i, x)
        return self._mab1(x, h)

ISAB = InducedSetAttentionBlock

class MultiheadAttentionPooling(nn.Module):
    def __init__(self, num_seeds, num_dims, initializer='xavier-uniform', *args, **kwargs):
        super(MultiheadAttentionPooling, self).__init__()
        self._s = nn.Parameter(torch.Tensor(num_seeds, num_dims))
        initializers.get(initializer)(self._s)
        self._mab = MAB(dim, dim, dim, num_heads,*args, **kwargs)

    def forward(self, x):
        sshape = [1] + list(self._s.size())
        s = self._s.view(ishape)
        return self._mab(s, x)

PMA = MultiheadAttentionPooling
MAP = MultiheadAttentionPooling

class GraphAttention(nn.Module):
    def __init__(self, cin, cout, dropout, act='leakyrelu'):
        super(GraphAttention, self).__init__()

        self.dropout = None
        if dropout is not None and dropout > 0 and dropout < 1:
            self.dropout = nn.Dropout(p=dropout)

        self.conv1 = snt.build_conv1d(cin,cout,ksize=1, bias=False, act=None, dropout=None)
        self.conv2 = snt.build_conv1d(2*cout, cout, ksize=1, bias=False, act='lrelu', dropout=None)
        self.act = activations.get(act)

    def neighboring(self, x, k=20):
        # distance: [batch-size, num-points, k]
        # index: [batch-size, num-points, k]
        index = snt.knn(t, k)[1]
        if x.is_cuda:
            device = torch.device('cuda:{}'.format(x.get_device()))
        else:
            device = torch.device('cpu')
        idx_base = torch.arange(0, batch_size, device=device).view(-1,1,1) * num_points
        idx = index + idx_base
        idx = idx.view(-1)

        x = x.transpose(2,1).contiguous()
        # y: [batch-size, num-points, k, dims]
        y = x.view(batch_size*num_points,-1)[idx,:].view(batch_size,num_points,k,-1)
        return y.permute(0,3,2,1)

    def forward(self, x):
        # x: [batch-size x num-dims x num-points]
        # y: [batch-size x num-dims' x num-points]
        y = self.conv1(x)
        # n: [batch-size x num-dims' x k x num-points]
        n = self.neighboring(y)
        # ny: [batch-size x 2num-dims' x k x num-points]
        ny = torch.cat([n,y.unsqueeze(2).repeat(1,1,k,1)],dim=2)
        # e: [batch-size x 1 x k x num-points]
        e = torch.softmax(self.conv2(ny),dim=2)
        # x: [batch-size x num-dims' x num-points]
        x = (e * n).sum(2)

class MultiheadGraphAttention(nn.Module):
    def __init__(self, cin, cout, num_heads, dropout, act='leakyrelu', aggregator='concat'):
        super(MultiheadGraphAttention, self).__init__()
        assert aggregator in ['concat', 'mean', 'max'], 'aggregator `{}` not support'.format(aggregator)
        if aggregator == 'concat':
            self.aggregator = self.concat_attention
        elif aggregator == 'mean':
            self.aggregator = self.mean_attention
        else:
            self.aggregator = self.max_attention
        single_cout = cout // num_heads
        self.gats = [GraphAttention(cin, single_cout, dropout, activate) for _ in range(num_heads)]
 
    def concat_attention(self, xs, dim=1):
        return F.elu(torch.cat(xs, dim=dim))

    def max_attention(self, xs, dim=1):
        y = torch.stack(xs, dim=dim)
        return y.max(y, dim=dim)[0]

    def mean_attention(self, xs, dim=1):
        y = torch.stack(xs, dim=dim)
        return y.mean(dim=dim)

    def forward(self,x):
        xs = [gat(x) for gat in self.gats]
        return self.aggregator(xs)



class ChannelAttentionBase(nn.Module):
    def __init__(self,
                 num_channels,
                 weight_mode='softmax',
                 **kwargs):
        super(ChannelAttentionBase, self).__init__()
        assert weight_mode in ['softmax', 'sigmoid'], 'weight mode `{}` not support'.format(weight_mode)
        if weight_mode == 'softmax':
            self.act = nn.Softmax(dim=1)
        else:
            self.act = nn.Sigmoid()
        self.num_channels = num_channels
        self.pool = None
        self.conv = None

    def forward(self, x):
        y = self.act(self.conv(self.pool(x)))
        return (y * x)

class ChannelAttention1d(ChannelAttentionBase):
    def __init__(self,*args,**kwargs):
        super(ChannelAttention1d, self).__init__(*args, **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = utils.build_conv1d(self.num_channels,self.num_channels,**kwargs)

class ChannelAttention2d(ChannelAttentionBase):
    def __init__(self,*args,**kwargs):
        super(ChannelAttention2d, self).__init__(*args, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = utils.build_conv2d(self.num_channels,self.num_channels,**kwargs)

class ChannelAttention3d(ChannelAttentionBase):
    def __init__(self,*args,**kwargs):
        super(ChannelAttention3d, self).__init__(*args, **kwargs)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = utils.build_conv3d(self.num_channels,self.num_channels,**kwargs)



if __name__ == '__main__':
    import numpy as np
    batch_size=1
    num_dims=6
    num_points=8
    #points = np.random.uniform(0,1,size=(batch_size,num_points,num_dims))
    points = np.random.uniform(0,1,size=(batch_size,num_dims,num_points))
    torch_points = torch.from_numpy(points).float()
    #attention = Attention(num_dims)
    ##attention = Attention()
    #att, w = attention(torch_points,torch_points,torch_points)
    #print('points:\n',points)
    ##print('w:\n',w.numpy())
    #print('attentions:\n',att.numpy())

    multi_att = MultiheadAttention(num_dims,3)
    results = multi_att(torch_points)

    #mab = MAB(num_dims, 3)
    #results = mab(torch_points)

    #sab = SAB(num_dims, num_heads=3)
    #results = sab(torch_points)

    #isab = ISAB(2, num_dims, num_heads=3)
    #results =isab(torch_points)
    print(results.detach().numpy())
    print(results.size())
