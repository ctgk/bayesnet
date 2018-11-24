import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Softplus(Function):

    @staticmethod
    def _forward(x):
        return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

    def backward(self, delta):
        x = self.args[0]
        dx = (np.tanh(0.5 * x.value) * 0.5 + 0.5) * delta
        x.backward(dx)


def softplus(x):
    """
    smoothed rectified linear unit

    log(1 + exp(x))
    """
    return Softplus().forward(x)
