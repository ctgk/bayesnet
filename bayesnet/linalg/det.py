from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Determinant(Function):

    def _forward(self, x):
        self._assert_ndim_atleast(x, 2)
        self.output = xp.linalg.det(x)
        return self.output

    def _backward(self, delta, x):
        dx = delta * self.output * xp.linalg.inv(xp.swapaxes(x, -1, -2))
        return dx


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
