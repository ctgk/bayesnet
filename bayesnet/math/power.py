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
        self.output = np.power(x, y)
        return self.output

    def _backward(self, delta, x, y):
        dx = y * np.power(x, y - 1) * delta
        if getattr(x, "size", 1) == 1:
            if x > 0:
                dy = self.output * np.log(x) * delta
            else:
                dy = None
        else:
            if (x > 0).all():
                dy = self.output * np.log(x) * delta
            else:
                dy = None
        return dx, dy


def power(x, y):
    """
    First array elements raised to powers from second array
    """
    return Power().forward(x, y)


def rpower(x, y):
    return Power().forward(y, x)
