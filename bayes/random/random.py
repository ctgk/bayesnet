import numpy as np


class RandomVariable(object):
    """
    base class for random variables
    """

    def nll(self, x):
        """
        negative log likelihood of the observation

        Parameters
        ----------
        x : (..., self.shape) np.ndarray
            observed data

        Returns
        -------
        nll : Tensor
            a value of negative log likelihood
        """
        self._check_input(x)
        if hasattr(self, "_nll"):
            return self._nll(x)
        else:
            raise NotImplementedError

    def pdf(self, x):
        """
        compute probability density function
        p(x|parameter)

        Parameters
        ----------
        x : (..., self.shape) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(x)
        if hasattr(self, "_pdf"):
            return self._pdf(x)
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
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self, x):
        assert isinstance(x, np.ndarray)
