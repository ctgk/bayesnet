import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.array.broadcast import broadcast


class Divide(Function):
    """
    divide arguments element-wise
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, y):
        return x / y

    @staticmethod
    def _backward(delta, x, y):
        dx = delta / y
        dy = -delta * x / y ** 2
        return dx, dy


def divide(x, y):
    return Divide().forward(x, y)


def rdivide(x, y):
    return Divide().forward(y, x)
