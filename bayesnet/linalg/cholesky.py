from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Cholesky(Function):

    def _forward(self, x):
        self.output = xp.linalg.cholesky(x)
        return self.output

    def _backward(self, delta, x):
        delta_lower = xp.tril(delta)
        P = phi(xp.einsum("...ij,...ik->...jk", self.output, delta_lower))
        S = xp.linalg.solve(
            xp.swapaxes(self.output, -1, -2),
            xp.einsum("...ij,...jk->...ik", P, xp.linalg.inv(self.output))
        )
        dx = S + xp.swapaxes(S, -1, -2) + xp.tril(xp.triu(S))
        return dx


def phi(x):
    return 0.5 * (xp.tril(x) + xp.tril(x, -1))


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
