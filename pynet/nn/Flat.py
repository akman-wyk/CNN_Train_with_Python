__all__ = ['Flat']


class Flat:
    """
    展平层，将卷积层的Tensor展平成FC层的矩阵
    """
    def __init__(self, in_shape, out_shape):
        self.in_c, self.in_h, self.in_w = in_shape
        self.out_len = out_shape

    def __call__(self, inputs, params, other):
        return self.forward(inputs)

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        assert inputs.shape[1:] == (self.in_c, self.in_h, self.in_w)
        return inputs.reshape(inputs.shape[0], -1), None

    def backward(self, grad_out, cache):
        assert len(grad_out.shape) == 2
        assert grad_out.shape[1] == self.out_len
        return grad_out.reshape(grad_out.shape[0], self.in_c, self.in_h, self.in_w)
