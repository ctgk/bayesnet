import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Trace(Function):

    def _forward(self, x):
        self._assert_ndim_equal_to(x, 2)
        return np.trace(x)

    @staticmethod
    def _backward(delta, x):
        dx = np.eye(x.shape[0], x.shape[1]) * delta
        return dx


def trace(x):
    """
    return sum of diagonal elements in two dimensional array

    Parameters
    ----------
    x : Tensor
        two dimensional array

    Returns
    -------
    Tensor
        sum of the diagonal elements
    """

    return Trace().forward(x)
