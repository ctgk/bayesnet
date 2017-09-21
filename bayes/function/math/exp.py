import numpy as np
from bayes.tensor.tensor import Tensor
from bayes.function.function import Function


class Exp(Function):

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.exp(x.value)
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = self.output * delta
        self.x.backward(dx)


def exp(x):
    """
    element-wise exponential function
    """
    return Exp().forward(x)
