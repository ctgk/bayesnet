import numpy as np
from bayesnet.array.broadcast import broadcast_to
from bayesnet.linalg.cholesky import cholesky
from bayesnet.linalg.det import det
from bayesnet.linalg.logdet import logdet
from bayesnet.linalg.solve import solve
from bayesnet.math.exp import exp
from bayesnet.math.sqrt import sqrt
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor


class MultivariateGaussian(RandomVariable):
    """
    Multivariate Gaussian distribution
    p(x|mu, cov)
    = exp{-0.5 * (x - mu)^T cov^-1 (x - mu)} * (1 / 2pi) ** (d / 2) * |cov^-1| ** 0.5
    where d = dimensionality

    Parameters
    ----------
    mu : (..., d) tensor_like
        mean parameter
    cov : (..., d, d) tensor_like
        variance-covariance matrix
    data : (..., d) tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu, cov, data=None, p=None):
        super().__init__(data, p)
        self.mu, self.cov = self._check_input(mu, cov)

    def _check_input(self, mu, cov):
        mu = self._convert2tensor(mu)
        cov = self._convert2tensor(cov)
        self._atleast_ndim(mu, 1)
        self._atleast_ndim(cov, 2)
        if cov.shape[-2:] != (mu.shape[-1], mu.shape[-1]):
            raise ValueError(
                "Mismatching dimensionality of mu and cov: {} and {}"
                .format(mu.shape[-1], cov.shape[-2:])
            )
        if mu.shape[:-1] != cov.shape[:-2]:
            shape = np.broadcast(mu.value[..., 0], cov.value[..., 0, 0]).shape
            if mu.shape[:-1] != shape:
                mu = broadcast_to(mu, shape + (mu.shape[-1],))
            if cov.shape[:-2] != shape:
                cov = broadcast_to(cov, shape + cov.shape[-2:])
        return mu, cov

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def cov(self):
        return self.parameter["cov"]

    @cov.setter
    def cov(self, cov):
        try:
            self.L = cholesky(cov)
        except np.linalg.LinAlgError:
            raise ValueError("cov must be positive-difinite matrix")
        self.parameter["cov"] = cov

    def forward(self):
        self.eps = np.random.normal(size=self.mu.shape)
        output = self.mu.value + np.einsum("...ij,...j->...i", self.L.value, self.eps)
        if isinstance(self.mu, Constant) and isinstance(self.cov, Constant):
            return Constant(output)
        return Tensor(output, self)

    def backward(self, delta):
        dmu = delta
        dL = delta * self.eps[:, None]
        self.mu.backward(dmu)
        self.L.backward(dL)

    def _pdf(self, x):
        d = x - self.mu
        d = d.reshape(*d.shape, 1)
        return (
            exp(-0.5 * (solve(self.cov, d) * d).sum(axis=(-2, -1)))
            / (2 * np.pi) ** (self.mu.shape[-1] * 0.5)
            / sqrt(det(self.cov))
        )

    def _log_pdf(self, x):
        d = x - self.mu
        d = d.reshape(*d.shape, 1)
        return (
            -0.5 * (solve(self.cov, d) * d).sum(axis=(-2, -1))
            - (self.mu.shape[-1] * 0.5) * np.log(2 * np.pi)
            - 0.5 * logdet(self.cov)
        )
