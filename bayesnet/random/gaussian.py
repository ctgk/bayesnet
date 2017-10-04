import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.function import Function
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

    def __init__(self, mu, std=None, var=None, data=None, prior=None):
        """
        construct Gaussian distribution

        Parameters
        ----------
        mu : tensor_like
            mean parameter
        std : tensor_like
            std parameter
        var : tensor_like
            variance parameter
        data : tensor_like
            observed data
        prior : RandomVariable
            prior distribution
        """
        super().__init__(data, prior)
        if std is not None and var is None:
            self.mu, self.std = self._check_input(mu, std)
        elif std is None and var is not None:
            self.mu, self.var = self._check_input(mu, var)
        elif std is None and var is None:
            raise ValueError("Either std or var must be assigned")
        else:
            raise ValueError("Cannot assign both std and var")

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        if x.shape != y.shape:
            shape = np.broadcast(x.value, y.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if y.shape != shape:
                y = broadcast_to(y, shape)
        return x, y

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def std(self):
        try:
            return self.parameter["std"]
        except KeyError:
            return sqrt(self.parameter["var"])

    @std.setter
    def std(self, std):
        try:
            ispositive = (std.value > 0).all()
        except AttributeError:
            ispositive = (std.value > 0)

        if not ispositive:
            raise ValueError("value of std must all be positive")
        self.parameter["std"] = std

    @property
    def var(self):
        try:
            return self.parameter["var"]
        except KeyError:
            return square(self.std)

    @var.setter
    def var(self, var):
        try:
            ispositive = (var.value > 0).all()
        except AttributeError:
            ispositive = (var.value > 0)

        if not ispositive:
            raise ValueError("value of var must all be positive")
        self.parameter["var"] = var

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
        self.std.backward(dstd)

    def _pdf(self, x):
        return (
            exp(-0.5 * square((x - self.mu) / self.std))
            / sqrt(2 * np.pi) / self.std
        )

    def _log_pdf(self, x):
        return GaussianLogPDF().forward(x, self.mu, self.var)


class GaussianLogPDF(Function):

    def _check_input(self, x, mu, var):
        x = self._convert2tensor(x)
        mu = self._convert2tensor(mu)
        var = self._convert2tensor(var)
        if not (x.shape == mu.shape == var.shape):
            shape = np.broadcast(x.value, mu.value, var.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if mu.shape != shape:
                mu = broadcast_to(mu, shape)
            if var.shape != shape:
                var = broadcast_to(var, shape)
        return x, mu, var

    def _forward(self, x, mu, var):
        x, mu, var = self._check_input(x, mu, var)
        self.x = x
        self.mu = mu
        self.var = var
        output = (
            -0.5 * np.square(x.value - mu.value) / var.value
            - 0.5 * np.log(var.value)
            - 0.5 * np.log(2 * np.pi)
        )
        return Tensor(output, function=self)

    def _backward(self, delta):
        dx = -0.5 * delta * (self.x.value - self.mu.value) / self.var.value
        dmu = -0.5 * delta * (self.mu.value - self.x.value) / self.var.value
        dvar = 0.5 * delta * (
            ((self.x.value - self.mu.value) / self.var.value) ** 2
            - 1 / self.var.value
        )
        self.x.backward(dx)
        self.mu.backward(dmu)
        self.var.backward(dvar)
