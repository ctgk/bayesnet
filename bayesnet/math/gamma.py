import scipy.special as sp
from bayesnet.function import Function
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Gamma(Function):

    def _forward(self, x):
        self.output = sp.gamma(x.value)
        return self.output

    def backward(self, delta):
        x = self.args[0]
        dx = delta * self.output * sp.digamma(x.value)
        x.backward(dx)


def gamma(x):
    """
    element-wise gamma function
    """
    return Gamma().forward(x)
