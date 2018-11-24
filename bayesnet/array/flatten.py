from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Flatten(Function):
    """
    flatten array
    """

    @classmethod
    def _forward(cls, x):
        cls._assert_ndim_atleast(x, 2)
        return x.flatten()

    @staticmethod
    def _backward(delta, x):
        return delta.reshape(*x.shape)


def flatten(x):
    """
    flatten N-dimensional array (N >= 2)
    """
    return Flatten().forward(x)
