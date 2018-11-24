import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Softmax(Function):

    def __init__(self, axis=-1):
        if not isinstance(axis, int):
            raise TypeError("axis must be int")
        self.axis = axis

    def _forward(self, x):
        y = x - np.max(x, self.axis, keepdims=True)
        np.exp(y, out=y)
        y /= y.sum(self.axis, keepdims=True)
        self.output = y
        return self.output

    def backward(self, delta):
        dx = self.output * delta
        dx -= self.output * dx.sum(self.axis, keepdims=True)
        self.args[0].backward(dx)


def softmax(x, axis=-1):
    """
    softmax function along specified axis
    y_k = exp(x_k) / sum_i(exp(x_i))
    """
    return Softmax(axis=axis).forward(x)
