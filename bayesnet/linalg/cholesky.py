import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Cholesky(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.linalg.cholesky(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        delta_lower = np.tril(delta)
        P = phi(np.einsum("...ij,...ik->...jk", self.output, delta_lower))
        S = np.linalg.solve(
            np.swapaxes(self.output, -1, -2),
            np.einsum("...ij,...jk->...ik", P, np.linalg.inv(self.output))
        )
        dx = S + np.swapaxes(S, -1, -2) + np.tril(np.triu(S))
        self.x.backward(dx)


def phi(x):
    return 0.5 * (np.tril(x) + np.tril(x, -1))


def cholesky(x):
    """
    cholesky decomposition of positive-definite matrix
    x = LL^T

    Parameters
    ----------
    x : (..., d, d) tensor_like
        positive-definite matrix

    Returns
    -------
    L : (..., d, d)
        cholesky decomposition
    """
    return Cholesky().forward(x)
