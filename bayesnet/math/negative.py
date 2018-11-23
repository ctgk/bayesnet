from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Negative(Function):
    """
    element-wise negative
    y = -x
    """

    @staticmethod
    def _forward(x):
        return -x.value

    def backward(self, delta):
        self.args[0].backward(-delta)


def negative(x):
    """
    element-wise negative
    """
    return Negative().forward(x)
