from bayesnet.function import Function


class RandomVariable(Function):
    """
    base class for random variables
    """

    def __init__(self, prior=None, name=None):
        """
        construct a random variable

        Parameters
        ----------
        prior : RandomVariable
            prior distribution
        name : str
            name of this random variable

        Returns
        -------
        parameter : dict
            dictionary of parameters
        """
        if not isinstance(name, str) and name is not None:
            raise TypeError("name must be str")
        self.name = name
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

    def pdf(self, x):
        """
        compute probability density function
        p(x|parameter)

        Parameters
        ----------
        x : (..., self.shape) np.ndarray
            observed data

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        if hasattr(self, "_pdf"):
            return self._pdf(x)
        else:
            raise NotImplementedError

    def log_pdf(self, x=None):
        """
        logarithm of probability density function

        Parameters
        ----------
        x : (..., self.shape) np.ndarray
            observed data

        Returns
        -------
        output : Tensor
            logarithm of probability density function
        """
        if hasattr(self, "_log_pdf"):
            return self._log_pdf(x)
        else:
            raise NotImplementedError
