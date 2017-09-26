from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Negative(Function):
    """
    element-wise negative
    y = -x
    """

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(-x.value)
        return Tensor(-x.value, function=self)

    def _backward(self, delta):
        dx = -delta
        self.x.backward(dx)


def negative(x):
    """
    element-wise negative
    """
    return Negative().forward(x)
