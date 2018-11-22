from contextlib import contextmanager
from bayesnet.random.random import RandomVariable
from bayesnet.tensor.parameter import Parameter


class Network(object):
    """
    a base class for network building

    Parameters
    ----------
    kwargs : tensor_like
        parameters to be optimized

    Attributes
    ----------
    parameter : dict
        dictionary of parameters to be optimized
    random_variable : dict
        dictionary of random varibles
    """

    def __init__(self):
        self._setting_parameter = False
        self._setting_prior_dist = False
        self.random_variable = {}
        self.parameter = {}

    @property
    def setting_parameter(self):
        return getattr(self, "_setting_parameter", False)

    @property
    def setting_prior_dist(self):
        return getattr(self, "_setting_prior_dist", False)

    @contextmanager
    def set_parameter(self):
        prev_scope = self._setting_parameter
        object.__setattr__(self, "_setting_parameter", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_setting_parameter", prev_scope)

    @contextmanager
    def set_prior_dist(self):
        prev_scope = self._setting_prior_dist
        object.__setattr__(self, "_setting_prior_dist", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_setting_prior_dist", prev_scope)

    def __setattr__(self, key, value):
        if isinstance(value, RandomVariable):
            if not self.setting_prior_dist:
                self.random_variable[key] = value
        elif self.setting_parameter:
            try:
                value = Parameter(value)
            except TypeError:
                raise TypeError(f"invalid type argument in self.set_parameter() scope: {type(value)}")
            self.parameter[key] = value
        object.__setattr__(self, key, value)

    def clear(self):
        """
        clear gradient and constructed bayesian network
        """
        for p in self.parameter.values():
            p.cleargrad()
        self.random_variable = {}

    def log_pdf(self, coef=1.):
        """
        compute logarithm of probabilty density function

        Parameters
        ----------
        coef : float
            coefficient to balance likelihood and prior
            assuming mini-batch size / whole data size for mini-batch training

        Returns
        -------
        logp : tensor_like
            logarithm of probability density function
        """
        logp = 0
        for rv in self.random_variable.values():
            if rv.observed:
                logp += rv.log_pdf().sum()
            else:
                logp += coef * rv.log_pdf().sum()
        return logp

    def elbo(self, coef=1.):
        """
        compute evidence lower bound of this model
        ln p(output) >= elbo

        Parameters
        ----------
        coef : float
            coefficient to balance likelihood and prior
            assuming mini-batch size / whole data size for mini-batch training

        Returns
        -------
        evidence : tensor_like
            evidence lower bound
        """
        evidence = 0
        for rv in self.random_variable.values():
            if rv.observed:
                evidence += rv.log_pdf().sum()
            else:
                evidence += -coef * rv.KLqp().sum()
        return evidence
