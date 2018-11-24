import numpy as np
from bayesnet.array.broadcast import broadcast
from bayesnet.function import Function
from bayesnet.math.log import log
from bayesnet.math.product import prod
from bayesnet.nonlinear.softmax import softmax
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.tensor import Tensor


class Categorical(RandomVariable):
    """
    Categorical distribution

    Parameters
    ----------
    mu : (..., K) tensor_like
        probability of each index
    logit : (..., K) tensor_like
        log-odd of each index
    axis : int
        axis along which represents each outcome
    data : tensor_like
        realization
    p : RandomVariable
        original distribution of a model

    Attributes
    ----------
    n_category : int
        number of categories
    """

    def __init__(self, mu=None, logit=None, axis=-1, data=None, p=None):
        super().__init__(data, p)
        assert axis == -1
        self.axis = axis
        if mu is not None and logit is None:
            self.mu = self._convert2tensor(mu)
        elif mu is None and logit is not None:
            self.logit = self._convert2tensor(logit)
        elif mu is None and logit is None:
            raise ValueError("Either mu or logit must not be None")
        else:
            raise ValueError("Cannot assign both mu and logit")

    @property
    def mu(self):
        try:
            return self.parameter["mu"]
        except KeyError:
            return softmax(self.parameter["logit"])

    @mu.setter
    def mu(self, mu):
        self._is_atleast_ndim(mu, 1)
        if not ((mu.value >= 0).all() and (mu.value <= 1).all()):
            raise ValueError("values of mu must be in [0, 1]")
        if not np.allclose(mu.value.sum(axis=self.axis), 1):
            raise ValueError(f"mu must be normalized along axis {self.axis}")
        self.parameter["mu"] = mu
        self.n_category = mu.shape[self.axis]

    @property
    def logit(self):
        try:
            return self.parameter["logit"]
        except KeyError:
            raise AttributeError("no attribute named logit")

    @logit.setter
    def logit(self, logit):
        self._is_atleast_ndim(logit, 1)
        self.parameter["logit"] = logit
        self.n_category = logit.shape[self.axis]

    def forward(self):
        if self.mu.ndim == 1:
            index = np.random.choice(self.n_category, p=self.mu.value)
            return np.eye(self.n_category)[index]
        elif self.mu.ndim == 2:
            indices = np.array(
                [np.random.choice(self.n_category, p=p.value) for p in self.mu.value]
            )
            return np.eye(self.n_category)[indices]
        else:
            raise NotImplementedError

    def _pdf(self, x):
        return prod(self.mu ** x, axis=self.axis)

    def _log_pdf(self, x):
        try:
            return -SoftmaxCrossEntropy(axis=self.axis).forward(self.logit, x)
        except AttributeError:
            return (x * log(self.mu)).sum(axis=self.axis)


class SoftmaxCrossEntropy(Function):

    enable_auto_broadcast = True

    def __init__(self, axis=-1):
        self.axis = axis

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    def _forward(self, x, t):
        self.y = np.exp(x - np.max(x, self.axis, keepdims=True))
        self.y /= np.sum(self.y, self.axis, keepdims=True)
        np.clip(self.y, 1e-10, 1, out=self.y)
        loss = -t * np.log(self.y)
        return loss

    def backward(self, delta):
        x, t = self.args[0], self.args[1]
        dx = delta * (self.y - t.value)
        dt = - delta * np.log(self.y)
        x.backward(dx)
        t.backward(dt)
