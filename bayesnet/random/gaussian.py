import numpy as np
from bayesnet.array.broadcast import broadcast
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

    Parameters
    ----------
    mu : tensor_like
        mean parameter
    std : tensor_like
        std parameter
    var : tensor_like
        variance parameter
    tau : tensor_like
        precision parameter
    data : tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu, std=None, var=None, tau=None, data=None, p=None):
        super().__init__(data, p)
        if std is not None and var is None and tau is None:
            self.mu, self.std = self._check_input(mu, std)
        elif std is None and var is not None and tau is None:
            self.mu, self.var = self._check_input(mu, var)
        elif std is None and var is None and tau is not None:
            self.mu, self.tau = self._check_input(mu, tau)
        elif std is None and var is None and tau is None:
            raise ValueError("Either std, var, or tau must be assigned")
        else:
            raise ValueError("Cannot assign more than two of these: std, var, tau")

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        [x, y] = broadcast([x, y])
        return x, y

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
            ispositive = (std.value > 0).all()
        except AttributeError:
            ispositive = (std.value > 0)

        if not ispositive:
            raise ValueError("value of std must all be positive")
        self.parameter["std"] = std

    @property
    def var(self):
        try:
            return self._var
        except AttributeError:
            return square(self.std)

    @var.setter
    def var(self, var):
        try:
            ispositive = (var.value > 0).all()
        except AttributeError:
            ispositive = (var.value > 0)

        if not ispositive:
            raise ValueError("value of var must all be positive")
        self._var = var
        self.parameter["std"] = sqrt(var)

    @property
    def tau(self):
        try:
            return self._tau
        except AttributeError:
            return 1 / square(self.std)

    @tau.setter
    def tau(self, tau):
        try:
            ispositive = (tau.value > 0).all()
        except AttributeError:
            ispositive = (tau.value > 0)

        if not ispositive:
            raise ValueError("value of tau must be positive")
        self._tau = tau
        self.parameter["std"] = 1 / sqrt(tau)

    def forward(self):
        self.eps = np.random.normal(size=self.mu.shape)
        output = self.mu.value + self.std.value * self.eps
        if isinstance(self.mu, Constant) and isinstance(self.var, Constant):
            return Constant(output)
        return Tensor(output, self)

    def backward(self, delta):
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
        return GaussianLogPDF().forward(x, self.mu, self.tau)


class GaussianLogPDF(Function):

    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, mu, tau):
        output = (
            -0.5 * np.square(x - mu) * tau
            + 0.5 * np.log(tau)
            - 0.5 * np.log(2 * np.pi)
        )
        return output

    @staticmethod
    def _backward(delta, x, mu, tau):
        dx = -0.5 * delta * (x - mu) * tau
        dmu = -dx
        dtau = 0.5 * delta * (
            1 / tau
            - (x - mu) ** 2
        )
        return dx, dmu, dtau
