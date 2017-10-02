from bayesnet.function import Function
from bayesnet.tensor.constant import Constant


class RandomVariable(Function):
    """
    base class for random variables
    """

    def __init__(self, data=None, prior=None):
        """
        construct a random variable

        Parameters
        ----------
        data : tensor_like
            observed data
        prior : RandomVariable
            prior distribution

        Returns
        -------
        parameter : dict
            dictionary of parameters
        observed : bool
            flag of observed or not
        """
        if data is not None and prior is not None:
            raise ValueError("Cannot assign both data and prior on a random variable")
        if data is not None:
            data = self._convert2tensor(data)
        self.data = data
        self.observed = isinstance(data, Constant)
        self.prior = prior
        self.parameter = dict()

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior):
        if prior is not None and not isinstance(prior, RandomVariable):
            raise TypeError("prior must be RandomVariable")
        self._prior = prior

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    def draw(self):
        """
        generate a sample

        Returns
        -------
        sample : tensor
            sample generated from this random variable
        """
        if self.observed:
            raise ValueError("draw method cannot be used for observed random variable")
        self.data = self.forward()
        return self.data

    def pdf(self, x=None):
        """
        compute probability density function
        p(x|parameter)

        Parameters
        ----------
        x : tensor_like
            observed data

        Returns
        -------
        p : Tensor
            value of probability density function for each input
        """
        if hasattr(self, "_pdf"):
            if x is None:
                return self._pdf(self.data)
            return self._pdf(x)
        else:
            raise NotImplementedError

    def log_pdf(self, x=None):
        """
        logarithm of probability density function

        Parameters
        ----------
        x : tensor_like
            observed data

        Returns
        -------
        output : Tensor
            logarithm of probability density function
        """
        if hasattr(self, "_log_pdf"):
            if x is None:
                return self._log_pdf(self.data)
            return self._log_pdf(x)
        else:
            raise NotImplementedError

    def KLqp(self, p=None):
        r"""
        compute Kullback Leibler Divergence
        KL(q(self)||p) = \int q(x) ln(q(x) / p(x)) dx

        Parameters
        ----------
        p : RandomVariable
            second argument of KL divergence

        Returns
        -------
        kl : Tensor
            KL divergence from this distribution to the given argument
        """
        if p is None:
            return self.log_pdf() - self.prior.log_pdf(self.data)
        return self.log_pdf() - p.log_pdf(self.data)
