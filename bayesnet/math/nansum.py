import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class NanSum(Function):
    """
    summation along given axis neglecting nan
    y = sum_i=1^N x_i
    """

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        return np.nansum(x, axis=self.axis, keepdims=self.keepdims)

    def _backward(self, delta, x):
        xdim, xshape = getattr(x, "ndim", 0), getattr(x, "shape", ())
        if isinstance(delta, np.ndarray) and (not self.keepdims) and (self.axis is not None):
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(xdim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
        dx = np.broadcast_to(delta, xshape) * (1 - np.isnan(x))
        return dx


def nansum(x, axis=None, keepdims=False):
    """
    returns summation of the elements along given axis neglecting nan
    y = sum_i=1^N x_i
    """
    return NanSum(axis=axis, keepdims=keepdims).forward(x)
