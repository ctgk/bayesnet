import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.array.broadcast import broadcast


class Subtract(Function):
    """
    subtract arguments element-wise
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, y):
        return x.value - y.value

    def backward(self, delta):
        self.args[0].backward(delta)
        self.args[1].backward(-delta)


def subtract(x, y):
    return Subtract().forward(x, y)


def rsubtract(x, y):
    return Subtract().forward(y, x)
