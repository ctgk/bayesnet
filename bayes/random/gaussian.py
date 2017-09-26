import numpy as np
from bayes.random.random import RandomVariable


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), sigma(std))
    = exp{-0.5 * (x - mu)^2 / sigma^2} / sqrt(2pi * sigma^2)
    """

    def __init__(self, mu, var, name=None):
        """
        construct Gaussian distribution

        Parameters
        ----------
        mu : int, float, RandomVariable
            mean parameter
        var : int, float, RandomVariable
            variance parameter
        """
        super().__init__(name=name)
        self.mu = mu
        self.var = var

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        try:
            mu = float(mu)
        except (TypeError, ValueError):
            pass

        if isinstance(mu, float):
            self.parameter["mu"] = mu
        elif isinstance(mu, RandomVariable):
            self.parameter["mu"] = mu
        else:
            raise TypeError(f"{type(mu)} is not acceptable for mu")

    @property
    def var(self):
        return self.parameter["var"]

    @var.setter
    def var(self, var):
        try:
            var = float(var)
        except TypeError:
            pass

        if isinstance(var, float):
            if var <= 0:
                raise ValueError("variance must be a positive value")
            self.parameter["var"] = var
        elif isinstance(var, RandomVariable):
            self.parameter["var"] = var
        else:
            raise TypeError(f"{type(var)} is not acceptable for var")

    @property
    def mean(self):
        if all(isinstance(p, float) for p in self.parameter.values()):
            return self.mu
        else:
            raise NotImplementedError

    @property
    def std(self):
        if all(isinstance(p, float) for p in self.parameter.values()):
            return self.var ** 0.5
        else:
            raise NotImplementedError

    @property
    def variance(self):
        if all(isinstance(p, float) for p in self.parameter.values()):
            return self.var
        else:
            raise NotImplementedError

    def _pdf(self, x):
        if isinstance(self.mu, RandomVariable):
            mu = self.mu.data
        else:
            mu = self.mu

        if isinstance(self.var, RandomVariable):
            var = self.var.data
        else:
            var = self.var

        return (
            np.exp(-0.5 * (x - mu) ** 2 / var)
            / np.sqrt(2 * np.pi * var)
        )

    def _log_pdf(self, x):
        if isinstance(self.mu, RandomVariable):
            mu = self.mu.data
        else:
            mu = self.mu

        if isinstance(self.var, RandomVariable):
            var = self.var.data
        else:
            var = self.var

        return (
            -0.5 * (x - mu) ** 2 / var
            - 0.5 * np.log(2 * np.pi * var)
        )

    def _draw(self, sample_size=1):
        if isinstance(self.mu, RandomVariable):
            loc = self.mu.draw(sample_size)
        else:
            loc = self.mu

        if isinstance(self.var, RandomVariable):
            scale = self.var.draw(sample_size) ** 0.5
        else:
            scale = self.var ** 0.5

        return np.random.normal(loc, scale, sample_size)

    def __neg__(self):
        return Gaussian(-self.mu, self.var)

    def __add__(self, x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            pass

        if isinstance(x, float):
            return Gaussian(self.mu + x, self.var)
        elif isinstance(x, Gaussian):
            return Gaussian(self.mu + x.mu, self.var + x.var)
        else:
            raise NotImplementedError

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return self.__add__(-x)

    def __rsub__(self, x):
        return -self.__sub__(x)

    def __mul__(self, x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            pass

        if isinstance(x, float):
            return Gaussian(self.mu * x, self.var * x ** 2)
        else:
            raise NotImplementedError

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        return self.__mul__(1 / x)
