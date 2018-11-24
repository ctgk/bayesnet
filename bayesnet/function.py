import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class Function(object):
    """
    Base class for differentiable functions
    """
    enable_auto_broadcast = False

    def forward(self, *args):
        args = [self._convert2tensor(arg) for arg in args]
        if self.enable_auto_broadcast:
            args = self._autobroadcast(args)
        self.args = args
        output = self._forward(*tuple(arg.value for arg in self.args))
        if all(isinstance(arg, Constant) for arg in self.args):
            return Constant(output)
        else:
            return Tensor(output, parent=self)

    def backward(self, delta):
        dargs = self._backward(delta, *tuple(arg.value for arg in self.args))
        if isinstance(dargs, tuple):
            for arg, darg in zip(self.args, dargs):
                arg.backward(darg)
        else:
            self.args[0].backward(dargs)

    @staticmethod
    def _autobroadcast(arg):
        raise NotImplementedError

    @staticmethod
    def _forward(*args):
        raise NotImplementedError

    @staticmethod
    def _backward(*args):
        raise NotImplementedError

    @staticmethod
    def _convert2tensor(x):
        if isinstance(x, (int, float, np.number, np.ndarray)):
            x = Constant(x)
        elif not isinstance(x, Tensor):
            raise TypeError(
                "Unsupported class for input: {}".format(type(x))
            )
        return x

    @staticmethod
    def _assert_ndim_equal_to(x, ndim):
        xdim = getattr(x, "ndim", 0)
        if xdim != ndim:
            raise ValueError(
                "dimensionality of the input must be {}, not {}"
                .format(ndim, xdim)
            )

    @staticmethod
    def _assert_ndim_atleast(x, ndim):
        xdim = getattr(x, "ndim", 0)
        if xdim < ndim:
            raise ValueError(
                "dimensionality of the input must be"
                " larger or equal to {}, not {}"
                .format(ndim, xdim)
            )
