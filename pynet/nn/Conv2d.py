# -*- coding: utf-8 -*-

# @Time    : 19-7-2 上午9:53
# @Author  : zj
import random

from .utils import *
from .im2row import *
from .pool2row import *
from .Layer import *

__all__ = ['Conv2d']


class Conv2d:
    """
    convolutional layer
    卷积层
    """

    def __init__(self, in_c, filter_h, filter_w, filter_num, stride=1, padding=0, weight_scale=1e-2, weight_loc=0.,
                 bias_value=0., cut_weight=False, cut_percent=0.9, cut_weight_mode='same'):
        """
        :param in_c: 输入数据体通道数
        :param filter_h: 滤波器长
        :param filter_w: 滤波器宽
        :param filter_num: 滤波器个数
        :param stride: 步长
        :param padding: 零填充
        :param weight_scale:
        """
        super(Conv2d, self).__init__()
        self.in_c = in_c
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_num = filter_num
        self.stride = stride
        self.padding = padding
        self.weight_scale = weight_scale
        self.weight_loc = weight_loc
        self.bias_value = bias_value

        self.cut_weight = cut_weight
        self.cut_percent = cut_percent
        self.cut_weight_mode = cut_weight_mode
        self.rand_data = None

        self.mode = 'train'

    def __call__(self, inputs, params, other):
        return self.forward(inputs, params)

    def forward(self, inputs, params):
        w, b = params

        if self.cut_weight is True and self.mode is 'train':
            self.rand_data = np.random.rand(*w.shape)

        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]
        out_h = int((H - self.filter_h + 2 * self.padding) / self.stride + 1)
        out_w = int((W - self.filter_w + 2 * self.padding) / self.stride + 1)

        a = im2row_indices(inputs, self.filter_h, self.filter_w, stride=self.stride, padding=self.padding)
        if self.cut_weight is True and self.mode is 'train':
            z = a.dot(cut_data(w, self.cut_percent, self.rand_data)) + b
        else:
            z = a.dot(w) + b

        out = conv_fc2output(z, N, out_h, out_w)
        cache = (a, inputs.shape, w, b)
        return out, cache

    def backward(self, grad_out, cache):
        assert len(grad_out.shape) == 4

        a, input_shape, w, b = cache

        if self.cut_weight is True and self.cut_weight_mode is not 'same':
            self.rand_data = np.random.rand(*w.shape)

        dz = conv_output2fc(grad_out)
        grad_W = a.T.dot(dz)
        grad_b = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.cut_weight is True:
            da = dz.dot(cut_data(w, self.cut_percent, self.rand_data).T)
        else:
            da = dz.dot(w.T)

        grad_in = row2im_indices(da, input_shape, field_height=self.filter_h,
                                 field_width=self.filter_w, stride=self.stride, padding=self.padding)
        return grad_in, grad_W, grad_b

    def get_params(self):
        return np.random.normal(loc=self.weight_loc, scale=self.weight_scale, size=(
            self.filter_h * self.filter_w * self.in_c, self.filter_num)), \
               np.ones((1, self.filter_num), dtype=float) * self.bias_value

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'test'
