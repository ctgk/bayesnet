import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Swapaxes(Function):

    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(np.swapaxes(x.value, self.axis1, self.axis2))
        return Tensor(np.swapaxes(x.value, self.axis1, self.axis2), parent=self)

    def backward(self, delta):
        dx = np.swapaxes(delta, self.axis2, self.axis1)
        self.x.backward(dx)


def swapaxes(x, axis1, axis2):
    """
    interchange two axes of an array

    Parameters
    ----------
    x : np.ndarray
        input array
    axis1: int
        first axis
    axis2: int
        second axis

    Returns
    -------
    output : np.ndarray
        interchanged array
    """
    return Swapaxes(axis1, axis2).forward(x)
