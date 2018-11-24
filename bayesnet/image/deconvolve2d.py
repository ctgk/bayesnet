from bayesnet import xp
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function
from bayesnet.image.util import img2patch, patch2img


class Deconvolve2d(Function):

    def __init__(self, stride, pad, shape):
        """
        construct 2 dimensional convolution function

        Parameters
        ----------
        stride : int or tuple of ints
            stride of kernel application
        pad : int or tuple of ints
            padding image
        shape : tuple of ints
            desired output image shape
        """
        self.stride = self._check_tuple(stride, "stride")
        self.pad = self._check_tuple(pad, "pad")
        self.pad = (0,) + self.pad + (0,)
        if shape is None:
            self.shape = None
        elif not isinstance(shape, tuple):
            raise TypeError(
                "Unsupported type for shape: {}"
                .format(type(shape))
            )
        else:
            self.shape = self._check_tuple(shape, "shape")

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
        self._assert_ndim_equal_to(x, 4)
        self._assert_ndim_equal_to(y, 4)
        if x.shape[3] != y.shape[2]:
            raise ValueError(
                "shapes {} and {} not aligned: {} (dim 3) != {} (dim 2)"
                .format(x.shape, y.shape, x.shape[3], y.shape[2])
            )

    def _forward(self, x, y):
        self._check_input(x, y)
        if self.shape is None:
            shape = tuple(
                s * (xlen - 1) + ylen
                for s, xlen, ylen in zip(self.stride, x.shape[1:], y.shape)
            )
        else:
            shape = self.shape
        patch = xp.tensordot(x, y, (3, 3))
        output = patch2img(
            patch,
            self.stride,
            (len(x),) + shape + (x.shape[-1],)
        )
        output = output[
            :,
            self.pad[1]: output.shape[1] - self.pad[1],
            self.pad[2]: output.shape[2] - self.pad[2]
        ]
        return output

    def _backward(self, delta, x, y):
        delta = xp.pad(delta, [(p,) for p in self.pad], "constant")
        dpatch = img2patch(delta, y.shape[:2], self.stride)
        dx = xp.tensordot(dpatch, y, axes=((3, 4, 5), (0, 1, 2)))
        dy = xp.tensordot(dpatch, x, axes=((0, 1, 2),) * 2)
        return dx, dy


def deconvolve2d(x, y, stride=1, pad=0, shape=None):
    """
    deconvolution of two tensors
    aka transposed convolution

    Parameters
    ----------
    x : (n_batch, xlen, ylen, in_channel) Tensor
        input tensor to be deconvolved
    y : (kx, ky, in_channel, out_channel) Tensor
        deconvolution kernel
    stride : int or tuple of ints (sx, sy)
        stride of kernel application
    pad : int or tuple of ints (px, py)
        padding image
    shape : tuple of ints (xlen', ylen')
        desired shape of output image
        If not specified, the output has the following length
        len' = s * (len - 1) - 2p + k

    Returns
    -------
    output : (n_batch, xlen', ylen', out_channel) Tensor
        The first argument deconvolved with the second one
        len' will be the following if not specified
        len' = s * (len - 1) - 2p + k
    """
    return Deconvolve2d(stride, pad, shape).forward(x, y)
