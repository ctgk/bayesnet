import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Determinant(Function):

    def _forward(self, x):
        self._is_atleast_ndim(x, 2)
        self.output = np.linalg.det(x.value)
        return self.output

    def backward(self, delta):
        x = self.args[0]
        dx = delta * self.output * np.linalg.inv(np.swapaxes(x.value, -1, -2))
        x.backward(dx)


def det(x):
    """
    determinant of a matrix

    Parameters
    ----------
    x : (..., d, d) tensor_like
        a matrix to compute its determinant

    Returns
    -------
    output : (...,) tensor_like
        determinant of the input matrix
    """
    return Determinant().forward(x)
