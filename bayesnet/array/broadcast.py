import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class BroadcastTo(Function):
    """
    Broadcast a tensor to an new shape
    """

    def __init__(self, shape):
        self.shape = shape

    def _forward(self, x):
        output = np.broadcast_to(x, self.shape)
        return output

    def backward(self, delta):
        x = self.args[0]
        dx = delta
        if delta.ndim != x.ndim:
            dx = dx.sum(axis=tuple(range(dx.ndim - x.ndim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(x.shape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        x.backward(dx)


def broadcast_to(x, shape):
    """
    Broadcast a tensor to an new shape
    """
    return BroadcastTo(shape).forward(x)


def broadcast(args):
    """
    broadcast list of tensors to make them have the same shape

    Parameters
    ----------
    args : list
        list of Tensor to be aligned

    Returns
    -------
    list
        list of Tensor whose shapes are aligned
    """
    shape = np.broadcast(*(arg.value for arg in args)).shape
    for i, arg in enumerate(args):
        if arg.shape != shape:
            args[i] = BroadcastTo(shape).forward(arg)
    return args
