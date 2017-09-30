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

    def __init__(self, **kwargs):
        self.random_variable = {}
        self.parameter = {}
        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self.parameter[key] = value
            else:
                try:
                    value = Parameter(value)
                except TypeError:
                    raise TypeError(f"invalid type argument: {type(value)}")
                self.parameter[key] = value
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        if isinstance(value, RandomVariable):
            if value.data is not None or value.prior is not None:
                self.random_variable[key] = value
        object.__setattr__(self, key, value)

    def cleargrad(self):
        for p in self.parameter.values():
            p.cleargrad()

    def elbo(self):
        """
        compute evidence lower bound of this model

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
                evidence += -rv.KLqp().sum()
        return evidence
