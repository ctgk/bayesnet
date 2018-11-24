from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Softplus(Function):

    @staticmethod
    def _forward(x):
        return xp.maximum(x, 0) + xp.log1p(xp.exp(-xp.abs(x)))

    @staticmethod
    def _backward(delta, x):
        dx = (xp.tanh(0.5 * x) * 0.5 + 0.5) * delta
        return dx


def softplus(x):
    """
    smoothed rectified linear unit

    log(1 + exp(x))
    """
    return Softplus().forward(x)
