import numpy as np
from bayesnet.config import Config
from bayesnet.function import Function


class BatchNormalization(Function):
    """
    Batch Normalization function
    """

    def __init__(self, momentum, init_mean, init_std, eps=1e-8):
        self.momentum = momentum
        self.mean = init_mean
        self.std = init_std
        self.eps = eps

    def _forward(self, x, scale=None, bias=None):
        if Config.is_training:
            self.x_mean = np.mean(x, axis=tuple(i for i in range(x.ndim - self.mean.ndim)))
            self.x_std = np.std(x, axis=tuple(i for i in range(x.ndim - self.std.ndim)))
            normalized = (x - self.x_mean) / (self.x_std + self.eps)
            self.mean = self.momentum * self.mean + (1 - self.momentum) * self.x_mean
            self.std = self.momentum * self.std + (1 - self.momentum) * self.x_std
        else:
            normalized = (x - self.mean) / (self.std + self.eps)

        self.normalized = normalized
        if scale is not None:
            normalized = normalized * scale
        if bias is not None:
            normalized = normalized + bias

        return normalized

    def _backward(self, delta, x, scale=None, bias=None):
        dinput = []

        if bias is not None:
            dbias = np.sum(delta, axis=tuple(i for i in range(delta.ndim - bias.ndim)))
            dinput.append(dbias)
        if scale is not None:
            dscale = delta * self.normalized
            dscale = np.sum(dscale, axis=tuple(i for i in range(dscale.ndim - scale.ndim)))
            dinput.insert(0, dscale)
            dnormalized = delta * scale
        else:
            dnormalized = delta

        dzeromean = dnormalized / (self.x_std + self.eps)

        dstd = dnormalized * (self.x_mean - x) / ((self.x_std + self.eps) ** 2)
        dstd = np.sum(dstd, axis=tuple(i for i in range(dstd.ndim - self.x_std.ndim)))
        dvar = 0.5 * dstd / (self.x_std + self.eps)
        dzeromean += (2.0 / len(x)) * (x - self.x_mean) * dvar
        dmean = np.sum(dzeromean, axis=tuple(i for i in range(dzeromean.ndim - self.x_mean.ndim)))
        dx = dzeromean - dmean / len(x)
        dinput.insert(0, dx)

        return tuple(dinput)
