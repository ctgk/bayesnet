import numpy as np
from bayesnet.config import Config
from bayesnet.function import Function


class BatchRenormalization(Function):
    """
    Batch Renormalization function
    """

    def __init__(self, momentum, init_mean, init_std, eps=1e-8):
        self.momentum = momentum
        self.mean = init_mean
        self.std = init_std
        self.eps = eps

    def _forward(self, x, scale=None, bias=None):
        if Config.is_training:
            self.x_mean = np.mean(x, axis=tuple(i for i in range(x.ndim - self.mean.ndim)))
            x_var = np.var(x, axis=tuple(i for i in range(x.ndim - self.std.ndim)))
            self.x_std = np.sqrt(x_var + self.eps)
        normalized = (x - self.mean) / self.eps
        self.normalized = normalized
        if scale:
            normalized = normalized * scale
        if bias:
            normalized = normalized + bias
        return normalized

    @staticmethod
    def sum_first_n_axes(x, n):
        return np.sum(x, axis=tuple(i for i in range(n)))

    def _backward(self, delta, x, scale=None, bias=None):
        dinput = []

        if bias:
            dbias = self.sum_first_n_axes(delta, delta.ndim - bias.ndim)
            dinput.append(dbias)
        if scale:
            dscale = delta * self.normalized
            dscale = self.sum_first_n_axes(dscale, dscale.ndim - scale.ndim)
            dinput.insert(0, dscale)
            dnormalized = delta * scale
        else:
            dnormalized = delta

        dzeromean = dnormalized / self.x_std

        dstd = dnormalized * ()
