import numpy as np
from bayesnet.optimizer.optimizer import Optimizer


class Momentum(Optimizer):
    """
    Momentum optimizer
    initialization
    v = 0
    update rule
    v = v * momentum - learning_rate * gradient
    param += v
    """

    def __init__(self, parameter, learning_rate, momentum=0.9):
        super().__init__(parameter, learning_rate)
        self.momentum = momentum
        self.inertia = []
        for p in self.parameter:
            self.inertia.append(np.zeros(p.shape))

    def update(self):
        self.increment_iteration()
        for p, inertia in zip(self.parameter, self.inertia):
            inertia *= self.momentum
            inertia -= self.learning_rate * p.grad
            p.value += inertia
