__all__ = ['Simple']


class Simple(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, params, other):
        return self.forward(inputs)

    def forward(self, inputs):
        return self.func(inputs), None

    def backward(self, grad_out, cache):
        return grad_out
