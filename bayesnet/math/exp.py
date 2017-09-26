import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Exp(Function):

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.exp(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = self.output * delta
        self.x.backward(dx)


def exp(x):
    """
    element-wise exponential function
    """
    return Exp().forward(x)
