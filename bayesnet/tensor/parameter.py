from bayesnet.tensor.tensor import Tensor


class Parameter(Tensor):
    """
    parameter to be optimized
    """

    def __init__(self, value):
        super().__init__(value, parent=None)
        # self.grad = None

    # def _backward(self, delta, **kwargs):
    #     if self.grad is None:
    #         self.grad = delta
    #     else:
    #         self.grad += delta

    def cleargrad(self):
        self.grad = None
