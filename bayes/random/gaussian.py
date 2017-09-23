import numpy as np
from bayes.function.math.log import log
from bayes.function.math.square import square
from bayes.random.random import RandomVariable
from bayes.tensor.tensor import Tensor


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), sigma(std))
    = exp{-0.5 * (x - mu)^2 / sigma^2} / sqrt(2pi * sigma^2)
    """

    def __init__(self, mu, sigma):
        """
        construct Gaussian distribution

        Parameters
        ----------
        mu : tensor_like
            mean of each element
        sigma : tensor_like
            std of each element
        """
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return (
            "Gaussian(\n"
            f"    mu={self.mu},\n"
            f"    sigma={self.sigma}\n)"
        )

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        try:
            mu = Tensor(mu)
        except TypeError:
            pass

        if isinstance(mu, Tensor):
            self._mu = mu
        else:
            raise TypeError(f"{type(mu)} is not acceptable for mu")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        try:
            sigma = Tensor(sigma)
        except TypeError:
            pass

        if isinstance(sigma, Tensor):
            self._sigma = sigma
        else:
            raise TypeError(f"{type(sigma)} is not acceptable for sigma")

    @property
    def ndim(self):
        return self._mu.ndim

    @property
    def size(self):
        return self._mu.size

    @property
    def shape(self):
        return self._mu.shape

    @property
    def mean(self):
        if isinstance(self._mu, Tensor):
            return self._mu.value
        else:
            raise NotImplementedError

    @property
    def var(self):
        if isinstance(self._sigma, Tensor):
            return self._sigma.value ** 2
        else:
            raise NotImplementedError

    def _nll(self, x):
        d = x - self._mu
        n = x.size / self._sigma.size
        return (
            0.5 * square(d / self._sigma).sum()
            + n * log(self._sigma)
        )

    def _pdf(self, x):
        d = x - self._mu.value
        return (
            np.exp(-0.5 * (d / self._sigma.value) ** 2)
            / np.sqrt(2 * np.pi * self._sigma.value ** 2)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self._mu.value,
            scale=self._sigma.value,
            size=(sample_size,) + self.shape
        )
