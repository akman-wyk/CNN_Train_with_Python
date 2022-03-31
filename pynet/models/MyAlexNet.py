import numpy as np

from pynet import nn
from .Net import Net

__all__ = ['MyAlexNet']

model_urls = {
    'myalexnet': ''
}


class MyAlexNet(Net):
    def __init__(self, dropout=0.5, weight_scale=0.01, cut_percent=0.9, cut_weight=False, cut_weight_mode='same',
                 cut_grad=False):
        super(MyAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 96, padding=1, weight_scale=weight_scale, cut_weight=cut_weight,
                               cut_percent=cut_percent, cut_weight_mode=cut_weight_mode)
        self.conv2 = nn.Conv2d(96, 3, 3, 256, padding=1, weight_scale=weight_scale, cut_weight=cut_weight,
                               cut_percent=cut_percent, cut_weight_mode=cut_weight_mode)
        self.conv3 = nn.Conv2d(256, 3, 3, 384, padding=1, weight_scale=weight_scale, cut_weight=cut_weight,
                               cut_percent=cut_percent, cut_weight_mode=cut_weight_mode)
        self.conv4 = nn.Conv2d(384, 3, 3, 384, padding=1, weight_scale=weight_scale, cut_weight=cut_weight,
                               cut_percent=cut_percent, cut_weight_mode=cut_weight_mode)
        self.conv5 = nn.Conv2d(384, 3, 3, 256, padding=1, weight_scale=weight_scale, cut_weight=cut_weight,
                               cut_percent=cut_percent, cut_weight_mode=cut_weight_mode)

        self.maxPool1 = nn.MaxPool(2, 2, 96, stride=2)
        self.maxPool2 = nn.MaxPool(2, 2, 256, stride=2)
        self.maxPool3 = nn.MaxPool(2, 2, 256, stride=2)

        self.fc1 = nn.FC(4096, 2048, weight_loc=0., weight_scale=weight_scale, bias_value=0.)
        self.fc2 = nn.FC(2048, 2048, weight_loc=0., weight_scale=weight_scale, bias_value=0.)
        self.fc3 = nn.FC(2048, 10, weight_loc=0., weight_scale=weight_scale, bias_value=0.)

        self.flat = nn.Flat((256, 4, 4), 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn = nn.BN()
        self.prune = nn.Prune(cut_grad=cut_grad, cut_percent=cut_percent)

        self.other_params = {'dropout_param': {'mode': 'train', 'p': dropout},
                             'bn1': {'mode': 'train'},
                             'bn2': {'mode': 'train'},
                             }
        self.params = self._get_params()

        self.layers = []
        self._add_layers([(self.prune, None, None),
                          (self.conv1, ('W1', 'b1'), None),
                          (self.relu, None, None),
                          (self.maxPool1, None, None),
                          (self.prune, None, None),
                          (self.conv2, ('W2', 'b2'), None),
                          (self.relu, None, None),
                          (self.maxPool2, None, None),
                          (self.prune, None, None),
                          (self.conv3, ('W3', 'b3'), None),
                          (self.relu, None, None),
                          (self.prune, None, None),
                          (self.conv4, ('W4', 'b4'), None),
                          (self.relu, None, None),
                          (self.prune, None, None),
                          (self.conv5, ('W5', 'b5'), None),
                          (self.relu, None, None),
                          (self.maxPool3, None, None),
                          (self.prune, None, None),
                          (self.flat, None, None),
                          (self.dropout, None, 'dropout_param'),
                          (self.fc1, ('W6', 'b6'), None),
                          (self.prune, None, None),
                          (self.bn, ('gamma1', 'beta1'), 'bn1'),
                          (self.relu, None, None),
                          (self.dropout, None, 'dropout_param'),
                          (self.fc2, ('W7', 'b7'), None),
                          (self.prune, None, None),
                          (self.bn, ('gamma2', 'beta2'), 'bn2'),
                          (self.relu, None, None),
                          (self.prune, None, None),
                          (self.fc3, ('W8', 'b8'), None)
                          ])

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, 3, 32, 32]
        assert len(inputs.shape) == 4
        assert inputs.shape[1:] == (3, 32, 32)

        feature = inputs
        for layer in self.layers:
            # print(layer['grad_keys'])
            params = None
            if layer['params_key'] is not None:
                params = [self.params[k] for k in layer['params_key']]

            other = None
            if layer['other_key'] is not None:
                other = self.other_params[layer['other_key']]

            feature, layer['cache'] = layer['layer'](feature, params, other)

        return feature

    def backward(self, grad_out):
        # grad_out.shape = [N, 10]
        assert len(grad_out.shape) == 2
        assert grad_out.shape[1] == 10

        grads = dict()
        feature_grad = grad_out
        for layer in reversed(self.layers):
            backward_out = layer['layer'].backward(feature_grad, layer['cache'])
            if layer['params_key'] is not None:
                feature_grad = backward_out[0]
                assert len(layer['params_key']) == len(backward_out) - 1
                for i in range(len(layer['params_key'])):
                    grads[layer['params_key'][i]] = backward_out[i + 1]
            else:
                feature_grad = backward_out

        return grads

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

    def _add_layers(self, layer_list):
        for layer, params_key, other_key in layer_list:
            self.layers.append({
                'layer': layer,
                'params_key': params_key,
                'cache': None,
                'other_key': other_key
            })

    def train(self):
        self.conv1.train()
        self.conv2.train()
        self.conv3.train()
        self.conv4.train()
        self.conv5.train()
        for v in self.other_params.values():
            v['mode'] = 'train'

    def eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.conv4.eval()
        self.conv5.eval()
        for v in self.other_params.values():
            v['mode'] = 'test'
