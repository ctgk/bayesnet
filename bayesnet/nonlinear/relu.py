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
        return x.value.clip(min=0)

    def backward(self, delta):
        x = self.args[0]
        dx = (x.value > 0) * delta
        x.backward(dx)


def relu(x):
    """
    Rectified Linear Unit
    y = max(x, 0)
    """
    return ReLU().forward(x)
