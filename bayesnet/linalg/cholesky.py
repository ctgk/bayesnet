import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Cholesky(Function):

    def _forward(self, x):
        self.output = np.linalg.cholesky(x)
        return self.output

    def _backward(self, delta, x):
        delta_lower = np.tril(delta)
        P = phi(np.einsum("...ij,...ik->...jk", self.output, delta_lower))
        S = np.linalg.solve(
            np.swapaxes(self.output, -1, -2),
            np.einsum("...ij,...jk->...ik", P, np.linalg.inv(self.output))
        )
        dx = S + np.swapaxes(S, -1, -2) + np.tril(np.triu(S))
        return dx


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
