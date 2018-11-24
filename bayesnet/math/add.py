from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.array.broadcast import broadcast


class Add(Function):
    """
    add arguments element-wise
    """
    enable_auto_broadcast = True

    @staticmethod
    def _autobroadcast(args):
        return broadcast(args)

    @staticmethod
    def _forward(x, y):
        return x + y

    @staticmethod
    def _backward(delta, x, y):
        return delta, delta


def add(x, y):
    return Add().forward(x, y)
