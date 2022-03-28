import numpy as np

from pynet import nn
from .Net import Net

__all__ = ['MyAlexNet']

model_urls = {
    'myalexnet': ''
}


class MyAlexNet(Net):
    def __init__(self, dropout=0.5, weight_scale=0.01):
        super(MyAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 96, stride=1, padding=1, weight_loc=0., weight_scale=weight_scale,
                               bias_value=0.)
        self.conv2 = nn.Conv2d(96, 3, 3, 256, stride=1, padding=1, weight_loc=0., weight_scale=weight_scale,
                               bias_value=0.)
        self.conv3 = nn.Conv2d(256, 3, 3, 384, stride=1, padding=1, weight_loc=0., weight_scale=weight_scale,
                               bias_value=0.)
        self.conv4 = nn.Conv2d(384, 3, 3, 384, stride=1, padding=1, weight_loc=0., weight_scale=weight_scale,
                               bias_value=0.)
        self.conv5 = nn.Conv2d(384, 3, 3, 256, stride=1, padding=1, weight_loc=0., weight_scale=weight_scale,
                               bias_value=0.)

        self.maxPool1 = nn.MaxPool(2, 2, 96, stride=2)
        self.maxPool2 = nn.MaxPool(2, 2, 256, stride=2)
        self.maxPool3 = nn.MaxPool(2, 2, 256, stride=2)

        self.fc1 = nn.FC(4096, 2048, weight_loc=0., weight_scale=weight_scale, bias_value=0.)
        self.fc2 = nn.FC(2048, 2048, weight_loc=0., weight_scale=weight_scale, bias_value=0.)
        self.fc3 = nn.FC(2048, 10, weight_loc=0., weight_scale=weight_scale, bias_value=0.)

        self.relu = nn.ReLU()

        self.dropout_param = {'mode': 'train', 'p': dropout}
        self.dropout = nn.Dropout()

        self.bn = nn.BN()
        self.bn_params = {
            'bn1': {'mode': 'train'},
            'bn2': {'mode': 'train'},
        }

        self.params = self._get_params()

        self.z1 = None
        self.z1_cache = None
        self.z2_cache = None
        self.z3 = None
        self.z3_cache = None
        self.z4_cache = None
        self.z5 = None
        self.z5_cache = None
        self.z6 = None
        self.z6_cache = None
        self.z7 = None
        self.z7_cache = None
        self.z8_cache = None
        self.z9_cache = None
        self.z10 = None
        self.z10_cache = None
        self.z11_cache = None
        self.z12 = None
        self.z12_cache = None
        self.z13_cache = None

        self.bn1 = None
        self.bn1_cache = None
        self.bn2 = None
        self.bn2_cache = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, 3, 32, 32]
        assert len(inputs.shape) == 4
        assert inputs.shape[1:] == (3, 32, 32)
        self.z1, self.z1_cache = self.conv1(inputs, self.params['W1'], self.params['b1'])
        # print('z1', self.z1)
        a1 = self.relu(self.z1)
        z2, self.z2_cache = self.maxPool1(a1)
        # print('z2', z2)

        self.z3, self.z3_cache = self.conv2(z2, self.params['W2'], self.params['b2'])
        # print('z3', self.z3)
        a3 = self.relu(self.z3)
        z4, self.z4_cache = self.maxPool2(a3)
        # print('z4', z4)

        self.z5, self.z5_cache = self.conv3(z4, self.params['W3'], self.params['b3'])
        # print('z5', self.z5)
        a5 = self.relu(self.z5)

        self.z6, self.z6_cache = self.conv4(a5, self.params['W4'], self.params['b4'])
        # print('z6', self.z6)
        a6 = self.relu(self.z6)

        self.z7, self.z7_cache = self.conv5(a6, self.params['W5'], self.params['b5'])
        # print('z7', self.z7)
        a7 = self.relu(self.z7)
        z8, self.z8_cache = self.maxPool3(a7)
        # print('z8', z8)

        # [N, 256, 4, 4] -> [N, 4096]
        assert len(z8.shape) == 4
        assert z8.shape[1:] == (256, 4, 4)
        z8 = z8.reshape(z8.shape[0], -1)

        z9, self.z9_cache = self.dropout(z8, self.dropout_param)
        # print('z9', z9)
        self.z10, self.z10_cache = self.fc1(z9, self.params['W6'], self.params['b6'])
        self.bn1, self.bn1_cache = self.bn(self.z10, self.params['gamma1'], self.params['beta1'], self.bn_params['bn1'])
        # print('z10', self.z10)
        a10 = self.relu(self.bn1)

        z11, self.z11_cache = self.dropout(a10, self.dropout_param)
        # print('z11', z11)
        self.z12, self.z12_cache = self.fc2(z11, self.params['W7'], self.params['b7'])
        self.bn2, self.bn2_cache = self.bn(self.z12, self.params['gamma2'], self.params['beta2'], self.bn_params['bn2'])
        # print('z12', self.z12)
        a12 = self.relu(self.bn2)

        z13, self.z13_cache = self.fc3(a12, self.params['W8'], self.params['b8'])
        # print('z13', z13)

        return z13

    def backward(self, grad_out):
        # grad_out.shape = [N, 10]
        assert len(grad_out.shape) == 2
        assert grad_out.shape[1] == 10

        grad = dict()

        grad['W8'], grad['b8'], da12 = self.fc3.backward(grad_out, self.z13_cache)

        dbn2 = self.relu.backward(da12, self.bn2)
        dz12, grad['gamma2'], grad['beta2'] = self.bn.backward(dbn2, self.bn2_cache)
        grad['W7'], grad['b7'], dz11 = self.fc2.backward(dz12, self.z12_cache)
        da10 = self.dropout.backward(dz11, self.z11_cache)

        dbn1 = self.relu.backward(da10, self.bn1)
        dz10, grad['gamma1'], grad['beta1'] = self.bn.backward(dbn1, self.bn1_cache)
        grad['W6'], grad['b6'], dz9 = self.fc1.backward(dz10, self.z10_cache)
        dz8 = self.dropout.backward(dz9, self.z9_cache)

        # [N, 4096] -> [N, 256, 4, 4]
        assert len(dz8.shape) == 2
        assert dz8.shape[1] == 4096
        dz8 = dz8.reshape(dz8.shape[0], 256, 4, 4)

        da7 = self.maxPool3.backward(dz8, self.z8_cache)
        dz7 = self.relu.backward(da7, self.z7)
        grad['W5'], grad['b5'], da6 = self.conv5.backward(dz7, self.z7_cache)

        dz6 = self.relu.backward(da6, self.z6)
        grad['W4'], grad['b4'], da5 = self.conv4.backward(dz6, self.z6_cache)

        dz5 = self.relu.backward(da5, self.z5)
        grad['W3'], grad['b3'], dz4 = self.conv3.backward(dz5, self.z5_cache)

        da3 = self.maxPool2.backward(dz4, self.z4_cache)
        dz3 = self.relu.backward(da3, self.z3)
        grad['W2'], grad['b2'], dz2 = self.conv2.backward(dz3, self.z3_cache)

        da1 = self.maxPool1.backward(dz2, self.z2_cache)
        dz1 = self.relu.backward(da1, self.z1)
        grad['W1'], grad['b1'], dz0 = self.conv1.backward(dz1, self.z1_cache)

        return grad

    def _get_params(self):
        params = dict()
        params['W1'], params['b1'] = self.conv1.get_params()
        params['W2'], params['b2'] = self.conv2.get_params()
        params['W3'], params['b3'] = self.conv3.get_params()
        params['W4'], params['b4'] = self.conv4.get_params()
        params['W5'], params['b5'] = self.conv5.get_params()

        params['W6'], params['b6'] = self.fc1.get_params()
        params['W7'], params['b7'] = self.fc2.get_params()
        params['W8'], params['b8'] = self.fc3.get_params()

        params['gamma1'], params['beta1'] = self.bn.get_params(2048)
        params['gamma2'], params['beta2'] = self.bn.get_params(2048)

        return params

    def train(self):
        self.dropout_param['mode'] = 'train'
        for k, v in self.bn_params:
            v['mode'] = 'train'

    def eval(self):
        self.dropout_param['mode'] = 'test'
        for k, v in self.bn_params:
            v['mode'] = 'test'
