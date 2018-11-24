import numpy as np
from bayesnet.array.broadcast import broadcast
from bayesnet.function import Function
from bayesnet.math.log import log
from bayesnet.nonlinear.sigmoid import sigmoid
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.tensor import Tensor


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu) = mu^x (1 - mu)^(1 - x)

    Parameters
    ----------
    mu : tensor_like
        probability of value 1
    logit : tensor_like
        log-odd of value 1
    data : tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu=None, logit=None, data=None, p=None):
        super().__init__(data, p)
        if mu is not None and logit is None:
            mu = self._convert2tensor(mu)
            self.mu = mu
        elif mu is None and logit is not None:
            logit = self._convert2tensor(logit)
            self.logit = logit
        elif mu is None and logit is None:
            raise ValueError("Either mu or logit must not be None")
        else:
            raise ValueError("Cannot assign both mu and logit")

    @property
    def mu(self):
        try:
            return self.parameter["mu"]
        except KeyError:
            return sigmoid(self.logit)

    @mu.setter
    def mu(self, mu):
        try:
            inrange = (0 <= mu.value <= 1)
        except ValueError:
            inrange = ((mu.value >= 0).all() and (mu.value <= 1).all())

        if not inrange:
            raise ValueError("value of mu must all be positive")
        self.parameter["mu"] = mu

    @property
    def logit(self):
        try:
            return self.parameter["logit"]
        except KeyError:
            raise AttributeError("no attribute named logit")

    @logit.setter
    def logit(self, logit):
        self.parameter["logit"] = logit

    def forward(self):
        return (np.random.uniform(size=self.mu.shape) < self.mu.value).astype(np.int)

    def _pdf(self, x):
        return self.mu ** x * (1 - self.mu) ** (1 - x)

    def _log_pdf(self, x):
        try:
            return -SigmoidCrossEntropy().forward(self.logit, x)
        except AttributeError:
            return x * log(self.mu) + (1 - x) * log(1 - self.mu)


class SigmoidCrossEntropy(Function):
    """
    sum of cross entropies for binary data
    logistic sigmoid
    y_i = 1 / (1 + exp(-x_i))
    cross_entropy_i = -t_i * log(y_i) - (1 - t_i) * log(1 - y_i)
    Parameters
    ----------
    x : ndarary
        input logit
    y : ndarray
        corresponding target binaries
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, t):
        # y = sigmoid(x)
        # np.clip(y, 1e-10, 1 - 1e-10, out=y)
        # return np.sum(-t * np.log(y) - (1 - t) * np.log(1 - y))
        loss = (
            np.maximum(x, 0)
            - t * x
            + np.log1p(np.exp(-np.abs(x)))
        )
        return loss

    @staticmethod
    def _backward(delta, x, t):
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        dx = delta * (y - t)
        dt = -delta * x
        return dx, dt
