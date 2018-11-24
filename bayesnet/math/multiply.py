import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.array.broadcast import broadcast


class Multiply(Function):
    """
    multiply arguments element-wise
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, y):
        return x * y

    def backward(self, delta):
        x, y = self.args[0], self.args[1]
        dx = y.value * delta
        dy = x.value * delta
        x.backward(dx)
        y.backward(dy)


def multiply(x, y):
    return Multiply().forward(x, y)
