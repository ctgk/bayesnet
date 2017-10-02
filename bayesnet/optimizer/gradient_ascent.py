from bayesnet.optimizer.optimizer import Optimizer


class GradientAscent(Optimizer):
    """
    gradient ascent optimizer
    parameter += learning_rate * gradient
    """

    def update(self):
        """
        update parameters to be optimized
        """
        self.increment_iteration()
        for p in self.parameter:
            p.value += self.learning_rate * p.grad
