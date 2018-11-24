import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class LogDeterminant(Function):

    @classmethod
    def _forward(cls, x):
        cls._assert_ndim_atleast(x, 2)
        sign, output = np.linalg.slogdet(x)
        if np.any(sign < 1):
            raise ValueError("The input matrix has to be positive-definite")
        return output

    def backward(self, delta):
        x = self.args[0]
        dx = (delta.T * np.linalg.inv(np.swapaxes(x.value, -1, -2)).T).T
        x.backward(dx)


def logdet(x):
    """
    log determinant of a matrix

    Parameters
    ----------
    x : (..., d, d) tensor_like
        a matrix to compute its log determinant

    Returns
    -------
    output : (...,) tensor_like
        log determinant of the input matrix
    """
    return LogDeterminant().forward(x)
