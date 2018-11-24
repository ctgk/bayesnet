from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Reshape(Function):
    """
    reshape array
    """

    def __init__(self, shape):
        self.shape = shape

    def _forward(self, x):
        self._assert_ndim_atleast(x, 1)
        return x.reshape(*self.shape)

    def backward(self, delta):
        x = self.args[0]
        dx = delta.reshape(*x.shape)
        x.backward(dx)


def reshape(x, shape):
    """
    reshape N-dimensional array (N >= 1)
    """
    return Reshape(shape).forward(x)


def reshape_method(x, *shape):
    return Reshape(shape).forward(x)
