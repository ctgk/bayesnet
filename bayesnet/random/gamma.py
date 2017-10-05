import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a(shape), b(rate))
    = a^b * x^(a - 1) * e^(-bx) / Gamma(a)

    Parameters
    ----------
    shape : tensor_like
        shape parameter
    rate : tensor_like
        rate parameter
    data : tensor_like
        realization
    prior : RandomVariable
        prior distribution
    """

    def __init__(self, shape, rate, data=None, prior=None):
        super().__init__(data, prior)
        shape, rate = self._check_input(shape, rate)
        self.shape = shape
        self.rate = rate

    def _check_input(self, shape, rate):
        shape = self._convert2tensor(shape)
        rate = self._convert2tensor(rate)
        if shape.shape != rate.shape:
            shape_ = np.broadcast(shape.value, rate.value).shape
            if shape.shape != shape_:
                shape = broadcast_to(shape, shape_)
            if rate.shape != shape_:
                rate = broadcast_to(rate, shape)
        return shape, rate

    @property
    def shape(self):
        return self.parameter["shape"]

    @shape.setter
    def shape(self, shape):
        try:
            ispositive = (shape.value > 0).all()
        except AttributeError:
            ispositive = (shape.value > 0)

        if not ispositive:
            raise ValueError("value of shape must be positive")
        self.parameter["shape"] = shape

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
        output = np.random.gamma(self.shape.value, 1 / self.rate.value)
        if isinstance(self.shape, Constant) and isinstance(self.rate, Constant):
            return Constant(output)
        return Tensor(output, function=self)

    def backward(self, delta):
        raise NotImplementedError
