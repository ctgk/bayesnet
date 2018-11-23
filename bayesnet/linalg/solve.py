import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Solve(Function):
    enable_auto_broadcast = True

    @classmethod
    def _autobroadcast(cls, args):
        a, b = args[0], args[1]
        cls._is_atleast_ndim(a, 2)
        cls._is_atleast_ndim(b, 2)
        if a.shape[-2:] != (b.shape[-2], b.shape[-2]):
            raise ValueError(
                "Mismatching dimensionality of a and b: {} and {}"
                .format(a.shape[-2:], b.shape[-2:])
            )
        if a.shape[:-2] != b.shape[:-2]:
            shape = np.broadcast(a.value[..., 0, 0], b.value[..., 0, 0]).shape
            if a.shape[:-2] != shape:
                a = broadcast_to(a, shape + a.shape[-2:])
            if b.shape[:-2] != shape:
                b = broadcast_to(b, shape + b.shape[-2:])
        return [a, b]

    def _forward(self, a, b):
        self.output = np.linalg.solve(a.value, b.value)
        return self.output

    def backward(self, delta):
        a, b = self.args[0], self.args[1]
        db = np.linalg.solve(np.swapaxes(a.value, -1, -2), delta)
        da = np.einsum("...ij,...kj->...ik", -db, self.output)
        a.backward(da)
        b.backward(db)


def solve(a, b):
    """
    solve a linear matrix equation
    ax = b

    Parameters
    ----------
    a : (..., d, d) tensor_like
        coefficient matrix
    b : (..., d, k) tensor_like
        dependent variable

    Returns
    -------
    output : (..., d, k) tensor_like
        solution of the equation
    """
    return Solve().forward(a, b)
