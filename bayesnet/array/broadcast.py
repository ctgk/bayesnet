from bayesnet import xp
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
        if not isinstance(x, xp.ndarray):
            x = xp.array(x)
        output = xp.broadcast_to(x, self.shape)
        return output

    @staticmethod
    def _backward(delta, x):
        dx = delta
        xdim = getattr(x, "ndim", 0)
        xshape = getattr(x, "shape", ())
        if delta.ndim != xdim:
            dx = dx.sum(axis=tuple(range(dx.ndim - xdim)))
            if isinstance(dx, xp.number):
                dx = xp.array(dx)
        axis = tuple(i for i, len_ in enumerate(xshape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        return dx


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
    shape = xp.broadcast(*(arg.value for arg in args)).shape
    for i, arg in enumerate(args):
        if arg.shape != shape:
            args[i] = BroadcastTo(shape).forward(arg)
    return args
