from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Reshape(Function):
    """
    reshape array
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        x = self._convert2tensor(x)
        self._is_atleast_ndim(x, 1)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(x.value.reshape(*self.shape))
        return Tensor(x.value.reshape(*self.shape), function=self)

    def backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def reshape(x, shape):
    """
    reshape N-dimensional array (N >= 1)
    """
    return Reshape(shape).forward(x)


def reshape_method(x, *shape):
    return Reshape(shape).forward(x)
