import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Inverse(Function):

    def _forward(self, x):
        self._is_equal_to_ndim(x, 2)
        self.output = np.linalg.inv(x)
        return self.output

    def backward(self, delta):
        dx = -self.output.T @ delta @ self.output.T
        self.args[0].backward(dx)


def inv(x):
    """
    inverse of a matrix

    Parameters
    ----------
    x : (d, d) tensor_like
        a matrix to be inverted

    Returns
    -------
    output : (d, d) tensor_like
        inverse of the input
    """
    return Inverse().forward(x)
