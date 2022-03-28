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


def get_random_float_np_array(total, size, scale):
    return np.array([random.uniform(scale[0], scale[1]) for _ in range(total)]).reshape(size)


def my_train():
    data_dict = cifar.get_CIFAR10_data('./cifar-10-batches-py1')
    # print(data_dict)

    model = models.MyAlexNet(0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.params, lr=0.1, momentum=0.9, nesterov=True)

    solver = pynet.Solver(model, data_dict, criterion, optimizer, batch_size=128, num_epochs=10, print_every=1)
    solver.train()
    with open('my_alex_net.model', 'wb') as f:
        pickle.dump(model, f)

    plt = Draw()
    plt(solver.loss_history)
    plt.multi_plot((solver.train_acc_history, solver.val_acc_history), ('train', 'val'),
                   title='准确率', xlabel='迭代/次', ylabel='准确率', save_path='acc.png')
    print('best_train_acc: %f; best_val_acc: %f' % (solver.best_train_acc, solver.best_val_acc))


def my_test():
    x_test, y_test = cifar.get_CIFAR10_test_data('./cifar-10-batches-py1')
    with open('my_alex_net.model', 'rb') as f:
        model = pickle.load(f)
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

    out, cache = conv1.forward(input_, w, b)
    print('out\n', out)

    grad_out = get_random_float_np_array(9, [1, 1, 3, 3], [-3, 3])
    grad_w, grad_b, grad_in = conv1.backward(grad_out, cache)
    print('grad_out\n', grad_out)
    print('grad_w\n', grad_w)
    print('grad_b\n', grad_b)
    print('grad_in\n', grad_in)


def test_speed(batch):
    alex_net = models.MyAlexNet()
    inputs = get_random_float_np_array(batch * 3 * 32 * 32, [batch, 3, 32, 32], [0., 1.])
    grad_out = get_random_float_np_array(batch * 10, [batch, 10], [-0.5, 0.5])

    t1 = time.perf_counter()
    for _ in range(1):
        alex_net(inputs)
        alex_net.backward(grad_out)
    t2 = time.perf_counter()
    print(t2 - t1)


if __name__ == '__main__':
    my_test()
    # test acc: 0.8162
