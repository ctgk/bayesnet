class RandomVariable(object):
    """
    base class for random variables
    """

    def __init__(self, name):
        """
        construct a random variable

        Parameters
        ----------
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
        self.data = None
        self.parameter = dict()

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            if isinstance(value, float):
                string += f"{key}={value}"
            else:
                string += f"{key}=" + value.__format__(depth=4)
            string += "\n"
        string += ")"
        return string

    def __format__(self, depth="4"):
        indent = 4
        depth = int(depth) + indent

        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * depth)
            if isinstance(value, float):
                string += f"{key}={value}"
            else:
                string += f"{key}=" + value.__format__(depth=depth)
            string += "\n"
        string += " " * (depth - indent)
        string += ")"
        return string

    def observe(self, x):
        """
        set observed value

        Parameters
        ----------
        x : array_like
            observed data
        """
        self.data = x

    def pdf(self, x=None):
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
            if x is None:
                if self.data is None:
                    raise ValueError("x must not be None")
                return self._pdf(self.data)
            else:
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
            if x is None:
                if self.data is None:
                    raise ValueError("x must not be None")
                return self._log_pdf(self.data)
            else:
                return self._log_pdf(x)
        else:
            raise NotImplementedError

    def draw(self, sample_size=1):
        """
        draw samples from the distribution

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, self.shape) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            sample = self._draw(sample_size)
            if sample.size == 1:
                return sample.item()
            return sample
        else:
            raise NotImplementedError
