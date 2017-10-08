import numpy as np
from bayesnet.math.gamma import gamma
from bayesnet.math.log import log
from bayesnet.math.product import prod
from bayesnet.math.sum import sum
from bayesnet.random.random import RandomVariable


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution

    Parameters
    ----------
    alpha : (..., K) tensor_like
        pseudo-count of each outcome
    axis : int
        axis along which represents each outcome
    data : tensor_like
        realization
    prior : RandomVariable
        prior distribution

    Attributes
    ----------
    n_category : int
        number of categories
    """

    def __init__(self, alpha, axis=-1, data=None, prior=None):
        super().__init__(data, prior)
        assert axis == -1
        self.axis = axis
        self.alpha = self._convert2tensor(alpha)

    @property
    def alpha(self):
        return self.parameter["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        self._atleast_ndim(alpha, 1)
        if (alpha.value <= 0).any():
            raise ValueError("alpha must all be positive")
        self.parameter["alpha"] = alpha

    def forward(self):
        if self.alpha.ndim == 1:
            return np.random.dirichlet(self.alpha.value)
        else:
            raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _pdf(self, x):
        return (
            gamma(self.alpha.sum(axis=self.axis))
            * prod(
                x ** (self.alpha - 1)
                / gamma(self.alpha),
                axis=self.axis
            )
        )

    def _log_pdf(self, x):
        return (
            log(gamma(self.alpha.sum(axis=self.axis)))
            + sum(
                (self.alpha - 1) * log(x) - log(gamma(self.alpha)),
                axis=self.axis
            )
        )
