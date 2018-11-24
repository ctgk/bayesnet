from bayesnet import xp
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

    @staticmethod
    def _backward(delta, x, y):
        dx = y * delta
        dy = x * delta
        return dx, dy


def multiply(x, y):
    return Multiply().forward(x, y)
