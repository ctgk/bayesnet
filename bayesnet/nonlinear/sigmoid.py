import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Sigmoid(Function):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """

    def _forward(self, x):
        self.output = np.tanh(x.value * 0.5) * 0.5 + 0.5
        return self.output

    def backward(self, delta):
        dx = self.output * (1 - self.output) * delta
        self.args[0].backward(dx)


def sigmoid(x):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """
    return Sigmoid().forward(x)
