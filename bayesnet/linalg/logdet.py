import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class LogDeterminant(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self._atleast_ndim(x, 2)
        sign, self.output = np.linalg.slogdet(x.value)
        if sign != 1:
            raise ValueError("The input matrix has to be positive-definite")
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = delta * np.linalg.inv(np.swapaxes(self.x.value, -1, -2))
        self.x.backward(dx)


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
