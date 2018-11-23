from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Flatten(Function):
    """
    flatten array
    """

    @classmethod
    def _forward(cls, x):
        cls._is_atleast_ndim(x, 2)
        return x.value.flatten()

    def backward(self, delta):
        x = self.args[0]
        dx = delta.reshape(*x.shape)
        x.backward(dx)


def flatten(x):
    """
    flatten N-dimensional array (N >= 2)
    """
    return Flatten().forward(x)
