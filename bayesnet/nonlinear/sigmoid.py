from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Sigmoid(Function):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """

    def _forward(self, x):
        self.output = xp.tanh(x * 0.5) * 0.5 + 0.5
        return self.output

    def _backward(self, delta, x):
        dx = self.output * (1 - self.output) * delta
        return dx


def sigmoid(x):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """
    return Sigmoid().forward(x)
