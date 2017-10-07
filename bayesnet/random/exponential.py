import numpy as np
from bayesnet.math.exp import exp
from bayesnet.math.log import log
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Exponential(RandomVariable):
    """
    Exponential distribution aka negative exponential distribution
    p(x|rate) = rate * exp(-rate * x)
    rate > 0

    Parameters
    ----------
    rate : tensor_like
        rate parameter
    data : tensor_like
        realization of this distribution
    prior : RandomVariable
        prior distribution
    """

    def __init__(self, rate, data=None, prior=None):
        super().__init__(data, prior)
        rate = self._convert2tensor(rate)
        self.rate = rate

    @property
    def rate(self):
        return self.parameter["rate"]

    @rate.setter
    def rate(self, rate):
        try:
            ispositive = (rate.value > 0).all()
        except AttributeError:
            ispositive = (rate.value > 0)

        if not ispositive:
            raise ValueError("value of rate must be positive")
        self.parameter["rate"] = rate

    def forward(self):
        eps = np.random.uniform(size=self.rate.shape)
        np.clip(eps, 1e-8, 1 - 1e-8, out=eps)
        eps = -np.log(eps)
        self.output = eps / self.rate.value
        if isinstance(self.rate, Constant):
            return Constant(self.output)
        return Tensor(self.output, self)

    def backward(self, delta):
        drate = -delta * self.output / self.rate.value
        self.rate.backward(drate)

    def _pdf(self, x):
        return self.rate * exp(-self.rate * x)

    def _log_pdf(self, x):
        return -self.rate * x + log(self.rate)
