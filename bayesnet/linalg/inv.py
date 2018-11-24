from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Inverse(Function):

    def _forward(self, x):
        self._assert_ndim_equal_to(x, 2)
        self.output = xp.linalg.inv(x)
        return self.output

    def _backward(self, delta, x):
        dx = -self.output.T @ delta @ self.output.T
        return dx


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
