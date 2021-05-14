import torch
from torch import nn

def DeformConv2d(nn.Module):
    def __init__(self,cin,cout,
                 ksize=3,
                 padding=1,
                 act=None,
                 bias=None):
        self.ksize = ksize
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(cin,cout,kernel_size=ksize,stride=ksize,padding=padding,bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        num_dims = offset.size(1) // 2

        # change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # codes below are written to make sure same results of MXNet implementation
        # you can remove them, and it won't influence the module's performance
        # -------------------------------------------------------------------------------------------------
        # offset_index = torch.cat([torch.arange(0, 2*num_dims, 2),  torch.arange(1,2*num_dims+1, 2)]).long()
        # offset_index = offset_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # offset = torch.gather(offset, dim=1, index=offset_index)
        # -------------------------------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2*num_dims, h, w)
        p = self._get_p(offset, dtype).contiguous()

        # (b, h, w, 2*num_dims)
        p = p.permute(0,2,3,1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :num_dims],0,x.size(2)-1), torch.clamp(q_lt[...,num_dims:],0,x.size(3)-1)],dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :num_dims],0,x.size(2)-1), torch.clamp(q_rb[...,num_dims:],0,x.size(3)-1)],dim=-1).long()
        q_lb = torch.cat([q_lt[...,:num_dims], q_rb[...,num_dims:]], -1)
        q_rt = torch.cat([q_rb[...,:num_dims], q_lt[...,num_dims:]], -1)

        # (b, h, w, num_dims)
        mask = torch.cat([p[...,:num_dims].lt(self.padding)+p[...,:num_dims].gt(x.size(2)-1-self.padding), p[...,num_dims:].lt(self.padding)+p[...,num_dims:].gt(x.size(3)-1-self.padding)],dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p-torch.floor(p))
        p = p * (1-mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[...,:num_dims],0,x.size(2)-1), torch.clamp(p[...,num_dims:],0,x.size(3)-1)],dim=-1)

        # bilinear kernel (b, h, w, num_dims)
        g_lt = (1+(q_lt[...,:num_dims].type_as(p) - p[...,:num_dims])) * (1+(q_lt[...,num_dims:].type_as(p) - p[...,num_dims:]))
        g_rb = (1-(q_rb[...,:num_dims].type_as(p) - p[...,:num_dims])) * (1-(q_rb[...,num_dims:].type_as(p) - p[...,num_dims:]))
        g_lb = (1+(q_lb[...,:num_dims].type_as(p) - p[...,:num_dims])) * (1+(q_lb[...,num_dims:].type_as(p) - p[...,num_dims:]))
        g_rt = (1-(q_rt[...,:num_dims].type_as(p) - p[...,:num_dims])) * (1-(q_rt[...,num_dims:].type_as(p) - p[...,num_dims:]))

        # (b, c, h, w, num_dims)
        x_q_lt = self._get_x_q(x, q_lt, num_dims)
        x_q_rb = self._get_x_q(x, q_rb, num_dims)
        x_q_lb = self._get_x_q(x, q_lb, num_dims)
        x_q_rt = self._get_x_q(x, q_rt, num_dims)

        # (b, c, h, w, num_dims)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.ksize)

        return self.conv(x_offset)

def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset