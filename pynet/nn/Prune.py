from my_import import np
from .utils import *

__all__ = ['Prune']


class Prune:
    def __init__(self, cut_grad, cut_percent):
        self.cut_grad = cut_grad
        self.cut_percent = cut_percent

    def __call__(self, inputs, params, other):
        return self.forward(inputs)

    def forward(self, inputs):
        return inputs, None

    def backward(self, grad_out, cache):
        if self.cut_grad is True:
            return cut_data(grad_out, self.cut_percent)
        else:
            return grad_out
