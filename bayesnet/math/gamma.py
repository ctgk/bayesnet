import scipy.special as sp
from bayesnet.function import Function
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Gamma(Function):

    def _forward(self, x):
        self.output = sp.gamma(x)
        return self.output

    def _backward(self, delta, x):
        dx = delta * self.output * sp.digamma(x)
        return dx


def gamma(x):
    """
    element-wise gamma function
    """
    return Gamma().forward(x)
