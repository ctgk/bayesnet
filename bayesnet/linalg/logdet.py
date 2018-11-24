from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class LogDeterminant(Function):

    @classmethod
    def _forward(cls, x):
        cls._assert_ndim_atleast(x, 2)
        sign, output = xp.linalg.slogdet(x)
        if xp.any(sign < 1):
            raise ValueError("The input matrix has to be positive-definite")
        return output

    @staticmethod
    def _backward(delta, x):
        dx = (delta.T * xp.linalg.inv(xp.swapaxes(x, -1, -2)).T).T
        return dx


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
