import random
import time

import pynet
from my_import import np
import os

import pynet.models as models

from pynet.vision.data import cifar
import pynet.nn as nn
import pynet.optim as optim
from pynet.vision import Draw
import pickle
import pynet.nn.im2row as im2row
import pynet.nn.utils as util
import argparse


def get_random_float_np_array(total, size, scale):
    return np.array([random.uniform(scale[0], scale[1]) for _ in range(total)]).reshape(size)


def my_train(batch_size, num_epochs, cut_percent, cut_weight_mode):
    data_dict = cifar.get_CIFAR10_data('./cifar-10-batches-py1')
    # print(data_dict)

    model = models.MyAlexNet(0.5, 0.1, cut_percent=cut_percent, cut_weight_mode=cut_weight_mode, cut_weight=True,
                             cut_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params, lr=0.01, momentum=0.9, nesterov=True)

    solver = pynet.Solver(model, data_dict, criterion, optimizer, batch_size=batch_size, num_epochs=num_epochs,
                          print_every=1)
    solver.train()
    with open('my_alex_net.model', 'wb') as f:
        pickle.dump(model, f)

    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))
    with open('acc.txt', 'w') as f:
        for i in range(len(solver.train_acc_history)):
            f.write(
                f'epoch: {i + 1}, train acc: {solver.train_acc_history[i]:.10}, val_acc: {solver.val_acc_history[i]:.10}\n')


def my_test(model_file='./my_alex_net.model'):
    x_test, y_test = cifar.get_CIFAR10_test_data('./cifar-10-batches-py1')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    test_solver = pynet.Solver(model)
    print(f'test acc: {test_solver.check_accuracy(x_test, y_test, batch_size=128)}')


def test():
    conv1 = nn.Conv2d(1, 3, 3, 1, padding=1)
    input_ = get_random_float_np_array(9, [1, 1, 3, 3], [-10, 10])
    w = get_random_float_np_array(9, [9, 1], [-3, 3])
    b = get_random_float_np_array(1, [1, 1], [-3, 3])
    print('input_\n', input_)
    print('w\n', w)
    print('b\n', b)

    out, cache = conv1.forward(input_, (w, b))
    print('out\n', out)

    grad_out = get_random_float_np_array(9, [1, 1, 3, 3], [-3, 3])
    grad_w, grad_b, grad_in = conv1.backward(grad_out, cache)
    print('grad_out\n', grad_out)
    print('grad_w\n', grad_w)
    print('grad_b\n', grad_b)
    print('grad_in\n', grad_in)


def test_speed(batch):
    alex_net = models.MyAlexNet(cut_grad=True, cut_weight=True, cut_weight_mode='not_same')
    inputs = get_random_float_np_array(batch * 3 * 32 * 32, [batch, 3, 32, 32], [0., 1.])
    grad_out = get_random_float_np_array(batch * 10, [batch, 10], [-0.5, 0.5])
    optimizer = optim.SGD(alex_net.params, lr=0.1, momentum=0.9, nesterov=True)

    t1 = time.perf_counter()
    for _ in range(1):
        alex_net(inputs)
        grad = alex_net.backward(grad_out)
        for k in grad.keys():
            print(k, grad[k].shape)
        optimizer.step(grad)
    t2 = time.perf_counter()
    print(t2 - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--cut_percent', type=float, default=0.9)
    parser.add_argument('--cut_weight_mode', type=str, default='same')
    parser.add_argument('--model_file', type=str, default='./my_alex_net.model')

    args = parser.parse_args()

    my_test(args.model_file)
    # my_train(args.batch_size, args.num_epochs, args.cut_percent, args.cut_weight_mode)
    # data = np.array([random.uniform(-1, 1) for _ in range(4)]).reshape(2, 2)
    # print(data)
    # print(util.cut_data(data, 0.7))

    # test_speed(100)

    # my_test()
    # test acc: 0.8162
