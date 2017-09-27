import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.math.exp import exp
from bayesnet.math.log import log
from bayesnet.math.sqrt import sqrt
from bayesnet.math.square import square
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), sigma(std))
    = exp{-0.5 * (x - mu)^2 / sigma^2} / sqrt(2pi * sigma^2)
    """

    def __init__(self, mu, std, prior=None, name=None):
        """
        construct Gaussian distribution

        Parameters
        ----------
        mu : tensor_like
            mean parameter
        std : tensor_like
            std parameter
        prior : RandomVariable
            prior distribution
        name : str
            name of this RandomVariable
        """
        super().__init__(prior, name)
        mu, std = self._check_input(mu, std)
        self.mu = mu
        self.std = std

    def _check_input(self, mu, std):
        mu = self._convert2tensor(mu)
        std = self._convert2tensor(std)
        if mu.shape != std.shape:
            shape = np.broadcast(mu.value, std.value).shape
            if mu.shape != shape:
                mu = broadcast_to(mu, shape)
            if std.shape != shape:
                std = broadcast_to(std, shape)
        return mu, std

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def std(self):
        return self.parameter["std"]

    @std.setter
    def std(self, std):
        try:
            ispositive = all(std.value > 0)
        except TypeError:
            ispositive = (std.value > 0)

        if not ispositive:
            raise ValueError("value of std must all be positive")
        self.parameter["std"] = std

    def _pdf(self, x):
        return (
            exp(-0.5 * square((x - self.mu) / self.std))
            / sqrt(2 * np.pi) / self.std
        )

    def _log_pdf(self, x):
        return (
            -0.5 * square((x - self.mu) / self.std)
            - log(self.std)
            - 0.5 * log(2 * np.pi)
        )

    def _forward(self):
        self.eps = np.random.normal(size=self.mu.shape)
        output = self.mu.value + self.std.value * self.eps
        if isinstance(self.mu, Constant) and isinstance(self.var, Constant):
            return Constant(output)
        return Tensor(output, self)

    def _backward(self, delta):
        dmu = delta
        dstd = delta * self.eps
        self.mu.backward(dmu)
        self.dstd.backward(dstd)


def gaussian(mu, sigma):
    return Gaussian(mu, sigma).forward()
