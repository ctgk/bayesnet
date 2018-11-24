from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Transpose(Function):

    def __init__(self, axes=None):
        self.axes = axes

    def _forward(self, x):
        if self.axes is not None:
            self._assert_ndim_equal_to(x, len(self.axes))
        return xp.transpose(x, self.axes)

    def _backward(self, delta, *args):
        if self.axes is None:
            return xp.transpose(delta)
        else:
            return xp.transpose(delta, xp.argsort(self.axes))


def transpose(x, axes=None):
    return Transpose(axes).forward(x)


def transpose_method(x, *args):
    if args == ():
        args = None
    return Transpose(args).forward(x)
