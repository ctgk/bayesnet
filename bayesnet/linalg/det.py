import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Determinant(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self._is_atleast_ndim(x, 2)
        self.output = np.linalg.det(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, parent=self)

    def backward(self, delta):
        dx = delta * self.output * np.linalg.inv(np.swapaxes(self.x.value, -1, -2))
        self.x.backward(dx)


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
