from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Sqrt(Function):
    """
    element-wise square root of the input
    y = sqrt(x)
    """

    def _forward(self, x):
        self.output = xp.sqrt(x)
        return self.output

    def _backward(self, delta, x):
        dx = 0.5 * delta / self.output
        return dx


def sqrt(x):
    """
    element-wise square root of the input
    y = sqrt(x)
    """
    return Sqrt().forward(x)
