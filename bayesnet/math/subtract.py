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
        return x - y

    @staticmethod
    def _backward(delta, x, y):
        return delta, -delta


def subtract(x, y):
    return Subtract().forward(x, y)


def rsubtract(x, y):
    return Subtract().forward(y, x)
