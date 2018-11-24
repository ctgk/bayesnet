from bayesnet import xp
from bayesnet.optimizer.optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer
    initialization
    G = 0
    update rule
    G += gradient ** 2
    param -= learning_rate * gradient / sqrt(G + eps)
    """

    def __init__(self, parameter, learning_rate=0.001, epsilon=1e-8):
        super().__init__(parameter, learning_rate)
        self.epsilon = epsilon
        self.G = []
        for p in self.parameter:
            self.G.append(xp.zeros(p.shape))

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for p, G in zip(self.parameter, self.G):
            if p.grad is None:
                continue
            grad = p.grad
            G += grad ** 2
            p.value -= self.learning_rate * grad / (xp.sqrt(G) + self.epsilon)
