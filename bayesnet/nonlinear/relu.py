from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class ReLU(Function):
    """
    Rectified Linear Unit
    y = max(x, 0)
    """

    @staticmethod
    def _forward(x):
        return x.clip(min=0)

    @staticmethod
    def _backward(delta, x):
        dx = (x > 0) * delta
        return dx


def relu(x):
    """
    Rectified Linear Unit
    y = max(x, 0)
    """
    return ReLU().forward(x)
