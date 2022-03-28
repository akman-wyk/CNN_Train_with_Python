# -*- coding: utf-8 -*-

# @Time    : 19-6-30 上午10:33
# @Author  : zj

from my_import import np
from .Layer import *

__all__ = ['ReLU']


class ReLU(object):

    def __call__(self, inputs, params, other):
        return self.forward(inputs)

    def forward(self, inputs):
        cache = (inputs, )
        return np.maximum(inputs, 0), cache

    def backward(self, grad_out, cache):
        inputs = cache[0]
        if inputs.shape != grad_out.shape:
            print(inputs.shape, grad_out.shape)
            assert inputs.shape == grad_out.shape

        grad_in = grad_out.copy()
        grad_in[inputs < 0] = 0
        return grad_in
