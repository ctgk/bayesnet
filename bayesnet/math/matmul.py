import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class MatMul(Function):
    """
    Matrix multiplication function
    """

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        self._is_atleast_ndim(x, 2)
        self._is_atleast_ndim(y, 2)
        if x.shape[-1] != y.shape[-2]:
            raise ValueError(
                "shapes {} and {} not aligned: {} (dim -1) != {} (dim -2)"
                .format(x.shape, y.shape, x.shape[-1], y.shape[-2])
            )
        if x.shape[:-2] != y.shape[:-2]:
            shape = np.broadcast(x.value[..., 0, 0], y.value[..., 0, 0]).shape
            if x.shape[:-2] != shape:
                x = broadcast_to(x, shape + x.shape[-2:])
            if y.shape[:-2] != shape:
                y = broadcast_to(y, shape + y.shape[-2:])
        return x, y

    def forward(self, x, y):
        x, y = self._check_input(x, y)
        self.x = x
        self.y = y
        if isinstance(self.x, Constant) and isinstance(self.y, Constant):
            return Constant(x.value @ y.value)
        return Tensor(x.value @ y.value, function=self)

    def backward(self, delta):
        dx = delta @ np.swapaxes(self.y.value, -1, -2)
        dy = np.swapaxes(self.x.value, -1, -2) @ delta
        self.x.backward(dx)
        self.y.backward(dy)


def matmul(x, y):
    return MatMul().forward(x, y)


def rmatmul(x, y):
    return MatMul().forward(y, x)
