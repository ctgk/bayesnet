from bayesnet import xp
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Product(Function):

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, tuple):
            axis = tuple(sorted(axis))
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        self.output = xp.prod(x, axis=self.axis, keepdims=True)
        if not self.keepdims:
            output = xp.squeeze(self.output)
            if output.size == 1:
                output = output.item()
        else:
            output = self.output
        return output

    def _backward(self, delta, x):
        if not self.keepdims and self.axis is not None:
            for ax in self.axis:
                delta = xp.expand_dims(delta, ax)
        dx = delta * self.output / x
        return dx


def prod(x, axis=None, keepdims=False):
    """
    product of all element in the array

    Parameters
    ----------
    x : tensor_like
        input array
    axis : int, tuple of ints
        axis or axes along which a product is performed
    keepdims : bool
        keep dimensionality or not

    Returns
    -------
    product : tensor_like
        product of all element
    """
    return Product(axis=axis, keepdims=keepdims).forward(x)
