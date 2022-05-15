from pynet import nn
from .Net import Net
import queue

__all__ = ['MyResNet']

model_urls = {
    'myresnet': ''
}


class LayerNode:
    def __init__(self, name, layer, params_keys=None, other_key=None):
        self.name = name
        self.layer = layer
        self.params_keys = params_keys
        self.other_key = other_key
        self.cache = None
        self.pred = set()
        self.next = set()

    def add_pred(self, pred):
        self.pred.add(pred)

    def add_next(self, _next):
        self.next.add(_next)


class MyResNet(Net):
    def __init__(self, weight_scale=0.01, cut_percent=0.9, cut_weight=False, cut_weight_mode='same'):
        self.first_layer_name = 'conv1'
        self.last_layer_name = 'fc'
        self.layers, self.params, self.other_params = {}, {}, {}

        self.conv_info, bn_info, maxPool_info, relu_info, fc_info = self._get_info()

        # conv layer
        for info in self.conv_info:
            layer = nn.Conv2d(info['shape'][0], info['shape'][1], info['shape'][2], info['shape'][3],
                              padding=info['padding'], weight_scale=weight_scale, cut_weight=cut_weight,
                              cut_weight_mode=cut_weight_mode, cut_percent=cut_percent)
            name = info['name']
            self.layers[name] = LayerNode(name, layer, params_keys=(f'{name}_W', f'{name}_b'))
            self.params[f'{name}_W'], self.params[f'{name}_b'] = layer.get_params()

        # bn layer
        bn = nn.MyBN()
        for info in bn_info:
            name = info['name']
            self.layers[name] = LayerNode(name, bn, other_key=f'{name}_other_params')
            self.other_params[f'{name}_other_params'] = {'mode': 'train'}

        # maxPool layer
        for info in maxPool_info:
            layer = nn.MaxPool(info['shape'][0], info['shape'][1], info['shape'][2], stride=info['stride'])
            name = info['name']
            self.layers[name] = LayerNode(name, layer)

        # relu layer
        relu = nn.ReLU()
        for info in relu_info:
            name = info['name']
            self.layers[name] = LayerNode(name, relu)

        # fc layer
        for info in fc_info:
            layer = nn.FC(info['shape'][0], info['shape'][1], weight_loc=0., weight_scale=weight_scale, bias_value=0.)
            name = info['name']
            self.layers[name] = LayerNode(name, layer, params_keys=(f'{name}_W', f'{name}_b'))
            self.params[f'{name}_W'], self.params[f'{name}_b'] = layer.get_params()

        # flat layer
        self.layers['flat'] = LayerNode('flat', nn.Flat((512, 1, 1), 512))

        self._link_layer_line(
            ['conv1', 'bn1', 'relu1', 'conv2', 'bn2', 'relu2', 'conv3', 'bn3', 'relu3', 'conv4', 'bn4', 'relu4',
             'conv5', 'bn5', 'relu5', 'conv7', 'bn7', 'maxPool2', 'relu7', 'conv9', 'bn9', 'relu8', 'conv10', 'bn10',
             'relu9', 'conv12', 'bn12', 'maxPool4', 'relu11', 'conv14', 'bn14', 'relu12', 'conv15', 'bn15', 'relu13',
             'conv17', 'bn17', 'maxPool6', 'relu15', 'conv19', 'bn19', 'relu16', 'conv20', 'bn20', 'relu17', 'maxPool7',
             'flat', 'fc'])
        self._link_layer_line(
            ['conv1', 'bn1', 'relu1', 'relu3', 'relu5', 'conv6', 'bn6', 'relu6', 'maxPool1', 'conv8', 'bn8', 'relu7',
             'relu9', 'conv11', 'bn11', 'relu10', 'maxPool3', 'conv13', 'bn13', 'relu11', 'relu13', 'conv16', 'bn16',
             'relu14', 'maxPool5', 'conv18', 'bn18', 'relu15', 'relu17', 'maxPool7', 'flat', 'fc'])

    def _link_layer_line(self, layer_line):
        for i in range(len(layer_line) - 1):
            self.layers[layer_line[i]].add_next(layer_line[i + 1])
            self.layers[layer_line[i + 1]].add_pred(layer_line[i])

    def forward(self, inputs):
        pred_num = {k: len(self.layers[k].pred) for k in self.layers.keys()}
        features = {k: None for k in self.layers.keys()}
        process_queue = queue.Queue()
        final_output = None

        process_queue.put(self.first_layer_name)
        pred_num[self.first_layer_name] -= 1
        features[self.first_layer_name] = inputs
        while not process_queue.empty():
            layer = self.layers[process_queue.get()]
            feature = features[layer.name]
            params = [self.params[k] for k in layer.params_keys] if layer.params_keys is not None else None
            other = self.other_params[layer.other_key] if layer.other_key is not None else None

            output, layer.cache = layer.layer(feature, params, other)
            if len(layer.next) is 0:
                final_output = output
            else:
                for name in layer.next:
                    if features[name] is None:
                        features[name] = output
                    else:
                        assert features[name].shape == output.shape
                        features[name] += output

                    pred_num[name] -= 1
                    if pred_num[name] is 0:
                        process_queue.put(name)

        return final_output

    def backward(self, grad_out):
        next_num = {k: len(self.layers[k].next) for k in self.layers.keys()}
        feature_grads = {k: None for k in self.layers.keys()}
        process_queue = queue.Queue()
        grads = {}

        process_queue.put(self.last_layer_name)
        next_num[self.last_layer_name] -= 1
        feature_grads[self.last_layer_name] = grad_out
        while not process_queue.empty():
            layer = self.layers[process_queue.get()]
            backward_out = layer.layer.backward(feature_grads[layer.name], layer.cache)
            if layer.params_keys is not None:
                feature_grad = backward_out[0]
                assert len(layer.params_keys) == len(backward_out) - 1
                for i in range(len(layer.params_keys)):
                    grads[layer.params_keys[i]] = backward_out[i + 1]
            else:
                feature_grad = backward_out
            for name in layer.pred:
                if feature_grads[name] is None:
                    feature_grads[name] = feature_grad
                else:
                    assert feature_grads[name].shape == feature_grad.shape
                    feature_grads[name] += feature_grad

                next_num[name] -= 1
                if next_num[name] is 0:
                    process_queue.put(name)

        return grads

    @staticmethod
    def _get_info():
        conv_info = [{'name': 'conv1', 'shape': (3, 3, 3, 64), 'padding': 1},
                     {'name': 'conv2', 'shape': (64, 3, 3, 64), 'padding': 1},
                     {'name': 'conv3', 'shape': (64, 3, 3, 64), 'padding': 1},
                     {'name': 'conv4', 'shape': (64, 3, 3, 64), 'padding': 1},
                     {'name': 'conv5', 'shape': (64, 3, 3, 64), 'padding': 1},
                     {'name': 'conv6', 'shape': (64, 3, 3, 128), 'padding': 1},
                     {'name': 'conv7', 'shape': (64, 1, 1, 128), 'padding': 0},
                     {'name': 'conv8', 'shape': (128, 3, 3, 128), 'padding': 1},
                     {'name': 'conv9', 'shape': (128, 3, 3, 128), 'padding': 1},
                     {'name': 'conv10', 'shape': (128, 3, 3, 128), 'padding': 1},
                     {'name': 'conv11', 'shape': (128, 3, 3, 256), 'padding': 1},
                     {'name': 'conv12', 'shape': (128, 1, 1, 256), 'padding': 0},
                     {'name': 'conv13', 'shape': (256, 3, 3, 256), 'padding': 1},
                     {'name': 'conv14', 'shape': (256, 3, 3, 256), 'padding': 1},
                     {'name': 'conv15', 'shape': (256, 3, 3, 256), 'padding': 1},
                     {'name': 'conv16', 'shape': (256, 3, 3, 512), 'padding': 1},
                     {'name': 'conv17', 'shape': (256, 1, 1, 512), 'padding': 0},
                     {'name': 'conv18', 'shape': (512, 3, 3, 512), 'padding': 1},
                     {'name': 'conv19', 'shape': (512, 3, 3, 512), 'padding': 1},
                     {'name': 'conv20', 'shape': (512, 3, 3, 512), 'padding': 1}]

        bn_info = [{'name': 'bn1', 'size': 64 * 32 * 32},
                   {'name': 'bn2', 'size': 64 * 32 * 32},
                   {'name': 'bn3', 'size': 64 * 32 * 32},
                   {'name': 'bn4', 'size': 64 * 32 * 32},
                   {'name': 'bn5', 'size': 64 * 32 * 32},
                   {'name': 'bn6', 'size': 128 * 32 * 32},
                   {'name': 'bn7', 'size': 128 * 32 * 32},
                   {'name': 'bn8', 'size': 128 * 16 * 16},
                   {'name': 'bn9', 'size': 128 * 16 * 16},
                   {'name': 'bn10', 'size': 128 * 16 * 16},
                   {'name': 'bn11', 'size': 256 * 16 * 16},
                   {'name': 'bn12', 'size': 256 * 16 * 16},
                   {'name': 'bn13', 'size': 256 * 8 * 8},
                   {'name': 'bn14', 'size': 256 * 8 * 8},
                   {'name': 'bn15', 'size': 256 * 8 * 8},
                   {'name': 'bn16', 'size': 512 * 8 * 8},
                   {'name': 'bn17', 'size': 512 * 8 * 8},
                   {'name': 'bn18', 'size': 512 * 4 * 4},
                   {'name': 'bn19', 'size': 512 * 4 * 4},
                   {'name': 'bn20', 'size': 512 * 4 * 4}]

        maxPool_info = [{'name': 'maxPool1', 'shape': (2, 2, 128), 'stride': 2},
                        {'name': 'maxPool2', 'shape': (2, 2, 128), 'stride': 2},
                        {'name': 'maxPool3', 'shape': (2, 2, 256), 'stride': 2},
                        {'name': 'maxPool4', 'shape': (2, 2, 256), 'stride': 2},
                        {'name': 'maxPool5', 'shape': (2, 2, 512), 'stride': 2},
                        {'name': 'maxPool6', 'shape': (2, 2, 512), 'stride': 2},
                        {'name': 'maxPool7', 'shape': (4, 4, 512), 'stride': 4}]

        relu_info = [{'name': f'relu{i}'} for i in range(1, 18)]

        fc_info = [{'name': 'fc', 'shape': (512, 10)}]

        return conv_info, bn_info, maxPool_info, relu_info, fc_info

    def train(self):
        for info in self.conv_info:
            self.layers[info['name']].layer.train()
        for v in self.other_params.values():
            v['mode'] = 'train'

    def eval(self):
        for info in self.conv_info:
            self.layers[info['name']].layer.eval()
        for v in self.other_params.values():
            v['mode'] = 'test'
