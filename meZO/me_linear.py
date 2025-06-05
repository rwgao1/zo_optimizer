import torch
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch
from torch.autograd import Function

import math


class linearUnified(Function):
    '''
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    '''
    @staticmethod
    def forward(ctx, x, w, b, k):
        '''
        forward propagation
        x: [minibatch, input feature]
        w: [input feature, output feature]
        b: [output feature]
        k: top-k in the backprop of meprop
        '''
        ctx.k = k
        ctx.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        ctx.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(ctx.add_buffer, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        x, w, b = ctx.saved_tensors
        k = ctx.k
        dx = dw = db = None

        if k > 0 and k < w.size(1):  # backprop with top-k selection
            _, inds = dy.abs().sum(0).topk(k)
            inds = inds.view(-1)
            pdy = dy.index_select(-1, inds)
            if ctx.needs_input_grad[0]:
                dx = torch.mm(pdy, w.index_select(-1, inds).t_())
            if ctx.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(
                    -1, inds, torch.mm(x.t(), pdy))
            if ctx.needs_input_grad[2]:
                db = torch.mv(dy.t(), ctx.add_buffer)
        else:  # backprop without top-k selection
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if ctx.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
            if ctx.needs_input_grad[2]:
                db = torch.mv(dy.t(), ctx.add_buffer)

        return dx, dw, db, None  # None for k


class linear(Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''
    @staticmethod
    def forward(ctx, x, w, b, k, sparse=True):
        '''
        forward propagation
        x: [minibatch, input feature]
        w: [input feature, output feature]
        b: [output feature]
        k: top-k in the backprop of meprop
        sparse: whether to use sparse matrix multiplication
        '''
        ctx.k = k
        ctx.sparse = sparse
        ctx.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        ctx.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(ctx.add_buffer, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b = ctx.saved_tensors
        k = ctx.k
        sparse = ctx.sparse
        dx = dw = db = None

        if k > 0 and k < w.size(1):  # backprop with top-k selection
            _, indices = dy.abs().topk(k)
            if sparse:
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(
                    0, dy.size()[0], device=dy.device).long().unsqueeze_(-1).repeat(1, k)
                indices_stacked = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.sparse_coo_tensor(indices_stacked, values, dy.size(), device=dy.device)
                if ctx.needs_input_grad[0]:
                    dx = torch.sparse.mm(pdy, w.t())
                if ctx.needs_input_grad[1]:
                    dw = torch.sparse.mm(pdy.t(), x).t()
            else:
                pdy = dy.new_zeros(dy.size()).scatter_(
                    -1, indices, dy.gather(-1, indices))
                if ctx.needs_input_grad[0]:
                    dx = torch.mm(pdy, w.t())
                if ctx.needs_input_grad[1]:
                    dw = torch.mm(x.t(), pdy)
        else:  # backprop without top-k selection
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if ctx.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)

        if ctx.needs_input_grad[2]:
            db = torch.mv(dy.t(), ctx.add_buffer)

        return dx, dw, db, None, None  # None for k and sparse
    

class meLinear(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, unified=False):
        super(meLinear, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.unified = unified

        self.w = Parameter(torch.Tensor(self.in_, self.out_))
        self.b = Parameter(torch.Tensor(self.out_))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.unified:
            return linearUnified.apply(x, self.w, self.b, self.k)
        else:
            return linear.apply(x, self.w, self.b, self.k)

    def __repr__(self):
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__,
                                              self.in_, self.out_, 'unified'
                                              if self.unified else '', self.k)