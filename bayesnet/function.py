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

    @staticmethod
    def _autobroadcast(arg):
        raise NotImplementedError

    @staticmethod
    def _forward(*args):
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
    def _is_equal_to_ndim(x, ndim):
        xdim = getattr(x, "ndim", 0)
        if xdim != ndim:
            raise ValueError(
                "dimensionality of the input must be {}, not {}"
                .format(ndim, xdim)
            )

    @staticmethod
    def _is_atleast_ndim(x, ndim):
        xdim = getattr(x, "ndim", 0)
        if xdim < ndim:
            raise ValueError(
                "dimensionality of the input must be"
                " larger or equal to {}, not {}"
                .format(ndim, xdim)
            )
