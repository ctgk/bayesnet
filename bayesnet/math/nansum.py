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

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        output = np.nansum(x.value, axis=self.axis, keepdims=self.keepdims)
        if isinstance(self.x, Constant):
            return Constant(output)
        return Tensor(output, function=self)

    def backward(self, delta):
        if isinstance(delta, np.ndarray) and (not self.keepdims) and (self.axis is not None):
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(self.x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
        dx = np.broadcast_to(delta, self.x.shape) * (1 - np.isnan(self.x.value))
        self.x.backward(dx)


def nansum(x, axis=None, keepdims=False):
    """
    returns summation of the elements along given axis neglecting nan
    y = sum_i=1^N x_i
    """
    return NanSum(axis=axis, keepdims=keepdims).forward(x)
