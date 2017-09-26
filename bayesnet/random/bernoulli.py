import numpy as np
from bayes.function.math.log import log
from bayes.random.random import RandomVariable
from bayes.tensor.tensor import Tensor


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu(prob)) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, prob):
        """
        construct Bernoulli distribution

        Parameters
        ----------
        param : dict
            dictionary of parameter
        prob : tensor_like
            prob of value 1 for each element
        """
        self.parameter = dict()
        self.prob = prob

    @property
    def prob(self):
        return self.parameter["prob"]

    @prob.setter
    def prob(self, prob):
        try:
            prob = Tensor(prob)
        except TypeError:
            pass

        if isinstance(prob, Tensor):
            self.parameter["prob"] = prob
        else:
            raise TypeError(f"{type(prob)} is not acceptable for prob")

    def __repr__(self):
        return (
            "Bernoulli(\n"
            f"    prob={self.prob}\n)"
        )

    @property
    def ndim(self):
        return self.prob.ndim

    @property
    def size(self):
        return self.prob.size

    @property
    def shape(self):
        return self.prob.shape

    @property
    def mean(self):
        if isinstance(self.prob, Tensor):
            return self.prob.value
        else:
            raise NotImplementedError

    @property
    def var(self):
        if isinstance(self.prob, Tensor):
            return self.mean * (1 - self.mean)
        else:
            raise NotImplementedError

    def _nll(self, x):
        return (- x * log(self.prob) - (1 - x) * log(1 - self.prob)).sum()

    def _pdf(self, X):
        return np.prod(
            self.mean ** X * (1 - self.mean) ** (1 - X)
        )

    def _draw(self, sample_size=1):
        return (
            self.mean > np.random.uniform(size=(sample_size,) + self.shape)
        ).astype(np.int)
