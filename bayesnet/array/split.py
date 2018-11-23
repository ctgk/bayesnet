import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Nth(Function):

    def __init__(self, n):
        self.n = n

    def forward(self, x):
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(x.value)
        return Tensor(x.value, parent=self)

    def backward(self, delta):
        self.x.backward(delta, n=self.n)


class Split(Function):

    def __init__(self, indices_or_sections, axis=-1):
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def forward(self, x):
        x = self._convert2tensor(x)
        self._is_atleast_ndim(x, 1)
        self.x = x
        output = np.split(x.value, self.indices_or_sections, self.axis)
        if isinstance(self.x, Constant):
            return tuple([Constant(out) for out in output])
        self.n_output = len(output)
        self.delta = [None for _ in output]
        return tuple([Tensor(out, parent=self) for out in output])

    def backward(self, delta, n):
        self.delta[n] = delta
        if all([d is not None for d in self.delta]):
            dx = np.concatenate(self.delta, axis=self.axis)
            self.x.backward(dx)


def split(x, indices_or_sections, axis=-1):
    output = Split(indices_or_sections, axis).forward(x)
    return tuple([Nth(i).forward(out) for i, out in enumerate(output)])
