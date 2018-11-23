import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.array.broadcast import broadcast


class Power(Function):
    """
    First array elements raised to powers from second array
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    def _forward(self, x, y):
        self.output = np.power(x.value, y.value)
        return self.output

    def backward(self, delta):
        x, y = self.args[0], self.args[1]
        dx = y.value * np.power(x.value, y.value - 1) * delta
        if x.size == 1:
            if x.value > 0:
                dy = self.output * np.log(x.value) * delta
            else:
                dy = None
        else:
            if (x.value > 0).all():
                dy = self.output * np.log(x.value) * delta
            else:
                dy = None
        x.backward(dx)
        y.backward(dy)


def power(x, y):
    """
    First array elements raised to powers from second array
    """
    return Power().forward(x, y)


def rpower(x, y):
    return Power().forward(y, x)
