import numpy as np
from bayesnet.config import Config
from bayesnet.function import Function
from bayesnet.tensor.tensor import Tensor


class Dropout(Function):

    def __init__(self, prob):
        """
        construct dropout function
        Parameters
        ----------
        prob : float
            probability of dropping the input value
        """
        if not isinstance(prob, float):
            raise TypeError(f"prob must be float value, not {type(prob)}")
        if prob < 0 or prob > 1:
            raise ValueError(f"{prob} is out of the range [0, 1]")
        self.prob = prob
        self.coef = 1 / (1 - prob)

    def forward(self, x, is_dropping=None):
        x = self._convert2tensor(x)
        if is_dropping or (is_dropping is None and Config.is_training):
            self.x = x
            self.mask = (np.random.rand(*x.shape) > self.prob) * self.coef
            return Tensor(x.value * self.mask, parent=self)
        else:
            return x

    def backward(self, delta):
        dx = delta * self.mask
        self.x.backward(dx)


def dropout(x, prob, is_dropping=None):
    """
    dropout function

    Parameters
    ----------
    x : Tensor
        input
    prob : float
        probability of dropping input values
    is_dropping : bool, optional
        flag to drop input values
        if False, the output is the same as the input
        (the default is None, which uses bayesnet.Config.is_training flag)

    Returns
    -------
    Tensor
        output tensor with some values dropped if is_dropping is True
    """
    return Dropout(prob).forward(x, is_dropping)
