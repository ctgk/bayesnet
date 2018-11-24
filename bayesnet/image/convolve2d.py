import numpy as np
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.image.util import img2patch, patch2img


class Convolve2d(Function):

    def __init__(self, stride, pad):
        """
        construct 2 dimensional convolution function

        Parameters
        ----------
        stride : int or tuple of ints
            stride of kernel application
        pad : int or tuple of ints
            padding image
        """
        self.stride = self._check_tuple(stride, "stride")
        self.pad = self._check_tuple(pad, "pad")
        self.pad = (0,) + self.pad + (0,)

    def _check_tuple(self, tup, name):
        if isinstance(tup, int):
            tup = (tup,) * 2
        if not isinstance(tup, tuple):
            raise TypeError(
                "Unsupported type for {}: {}".format(name, type(tup))
            )
        if len(tup) != 2:
            raise ValueError(
                "the length of {} must be 2, not {}".format(name, len(tup))
            )
        if not all([isinstance(n, int) for n in tup]):
            raise TypeError(
                "Unsuported type for {}".format(name)
            )
        if not all([n >= 0 for n in tup]):
            raise ValueError(
                "{} must be non-negative values".format(name)
            )
        return tup

    def _check_input(self, x, y):
        self._is_equal_to_ndim(x, 4)
        self._is_equal_to_ndim(y, 4)
        if x.shape[3] != y.shape[2]:
            raise ValueError(
                "shapes {} and {} not aligned: {} (dim 3) != {} (dim 2)"
                .format(x.shape, y.shape, x.shape[3], y.shape[2])
            )

    def _forward(self, x, y):
        self._check_input(x, y)
        img = np.pad(x, [(p,) for p in self.pad], "constant")
        self.shape = img.shape
        self.patch = img2patch(img, y.shape[:2], self.stride)
        return np.tensordot(self.patch, y, axes=((3, 4, 5), (0, 1, 2)))

    def backward(self, delta):
        x, y = self.args[0], self.args[1]
        dx = patch2img(
            np.tensordot(delta, y.value, (3, 3)),
            self.stride,
            self.shape
        )
        slices = tuple(slice(p, len_ - p) for p, len_ in zip(self.pad, self.shape))
        dx = dx[slices]
        dy = np.tensordot(self.patch, delta, axes=((0, 1, 2),) * 2)
        x.backward(dx)
        y.backward(dy)


def convolve2d(x, y, stride=1, pad=0):
    """
    returns convolution of two tensors

    Parameters
    ----------
    x : (n_batch, xlen, ylen, in_channel) Tensor
        input tensor to be convolved
    y : (kx, ky, in_channel, out_channel) Tensor
        convolution kernel
    stride : int or tuple of ints (sx, sy)
        stride of kernel application
    pad : int or tuple of ints (px, py)
        padding image

    Returns
    -------
    output : (n_batch, xlen', ylen', out_channel) Tensor
        input convolved with kernel
        len' = (len + 2p - k) // s + 1
    """
    return Convolve2d(stride, pad).forward(x, y)
