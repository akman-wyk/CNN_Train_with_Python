# -*- coding: utf-8 -*-

# @Time    : 19-6-29 下午8:33
# @Author  : zj

from my_import import np
import time

__all__ = ['Solver']


class Solver(object):

    def __init__(self, model, data=None, criterion=None, optimizer=None, **kwargs):
        self.model = model
        if data is not None:
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_val = data['X_val']
            self.y_val = data['y_val']
        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_scheduler = kwargs.pop('lr_scheduler', None)
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.reg = kwargs.pop('reg', 1e-3)
        self.use_reg = self.reg != 0

        self.print_every = kwargs.pop('print_every', 1)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('未识别参数: %s' % extra)

        self.current_epoch = 0
        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _reset(self):
        self.current_epoch = 0
        self.best_train_acc = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _step(self, X_batch, y_batch):
        scores = self.model.forward(X_batch)
        # print(scores)
        loss, probs = self.criterion.forward(scores, y_batch)
        if self.use_reg:
            for k in self.model.params.keys():
                if 'W' in k:
                    loss += 0.5 * self.reg * np.sum(self.model.params[k] ** 2)

        grad_out = self.criterion.backward(probs, y_batch)
        grad = self.model.backward(grad_out)
        if self.use_reg:
            for k in grad.keys():
                if 'W' in k:
                    grad[k] += self.reg * self.model.params[k]

        self.optimizer.step(grad)

        return loss

    def check_accuracy(self, X, y, num_samples=None, batch_size=8):
        """
        精度测试，如果num_samples小于X长度，则从X中采样num_samples个图片进行检测
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        if N < batch_size:
            batch_size = N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.forward(X[start:end])
            y_pred.extend(np.argmax(scores, axis=1))
        acc = np.mean(np.array(y_pred).reshape(y.shape) == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)

        for i in range(self.num_epochs):
            self.current_epoch = i + 1
            if self.current_epoch == 60:
                self.optimizer.set_lr(0.001)
            start = time.time()
            total_loss = 0.
            self.model.train()
            for j in range(iterations_per_epoch):
                idx_start = j * self.batch_size
                idx_end = (j + 1) * self.batch_size
                X_batch = self.X_train[idx_start:idx_end]
                y_batch = self.y_train[idx_start:idx_end]

                loss = self._step(X_batch, y_batch)
                if not np.isnan(loss):
                    total_loss += loss
                # print(loss, end='  ')
            end = time.time()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.current_epoch % self.print_every == 0:
                avg_loss = total_loss / iterations_per_epoch
                self.loss_history.append(float('%.6f' % avg_loss))
                print(f'epoch: {self.current_epoch}, lr: {self.optimizer.get_lr()} time: {end - start} loss: {avg_loss:.10}')

                self.model.eval()
                train_acc = self.check_accuracy(self.X_train, self.y_train, batch_size=self.batch_size)
                val_acc = self.check_accuracy(self.X_val, self.y_val, batch_size=self.batch_size)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                print(f'train acc: {train_acc:.10}; val_acc: {val_acc:.10}')

                if val_acc >= self.best_val_acc and train_acc > self.best_train_acc:
                    self.best_train_acc = train_acc
                    self.best_val_acc = val_acc
                    self.best_params = dict()
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
