import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import bayesnet as bn


class BayesianNetwork(bn.Network):

    def __init__(self, n_input, n_hidden, n_output=1):
        super().__init__()
        with self.set_parameter(), self.set_prior_dist():
            self.w1_mu = np.zeros((n_input, n_hidden))
            self.w1_s = np.zeros((n_input, n_hidden))
            self.b1_mu = np.zeros(n_hidden)
            self.b1_s = np.zeros(n_hidden)
            self.w2_mu = np.zeros((n_hidden, n_hidden))
            self.w2_s = np.zeros((n_hidden, n_hidden))
            self.b2_mu = np.zeros(n_hidden)
            self.b2_s = np.zeros(n_hidden)
            self.w3_mu = np.zeros((n_hidden, n_output))
            self.w3_s = np.zeros((n_hidden, n_output))
            self.b3_mu = np.zeros(n_output)
            self.b3_s = np.zeros(n_output)
            self.p = bn.random.Gaussian(mu=0, std=1)

    def __call__(self, x, y=None):
        self.qw1 = bn.random.Gaussian(
            mu=self.w1_mu,
            std=bn.softplus(self.w1_s),
            p=self.p
        )
        self.qb1 = bn.random.Gaussian(
            mu=self.b1_mu,
            std=bn.softplus(self.b1_s),
            p=self.p
        )
        self.qw2 = bn.random.Gaussian(
            mu=self.w2_mu,
            std=bn.softplus(self.w2_s),
            p=self.p
        )
        self.qb2 = bn.random.Gaussian(
            mu=self.b2_mu,
            std=bn.softplus(self.b2_s),
            p=self.p
        )
        self.qw3 = bn.random.Gaussian(
            mu=self.w3_mu,
            std=bn.softplus(self.w3_s),
            p=self.p
        )
        self.qb3 = bn.random.Gaussian(
            mu=self.b3_mu,
            std=bn.softplus(self.b3_s),
            p=self.p
        )
        h = bn.tanh(x @ self.qw1.draw() + self.qb1.draw())
        h = bn.tanh(h @ self.qw2.draw() + self.qb2.draw())
        self.py = bn.random.Bernoulli(
            logit=h @ self.qw3.draw() + self.qb3.draw(),
            data=y
        )
        return self.py.mu.value


def main(visualize):
    x_train, y_train = make_moons(n_samples=500, noise=0.2)
    y_train = y_train[:, None]

    model = BayesianNetwork(2, 5, 1)
    optimizer = bn.optimizer.Adam(model, 0.1)
    optimizer.set_decay(0.9, 100)

    for _ in range(2000):
        model.clear()
        model(x_train, y_train)
        elbo = model.elbo()
        elbo.backward()
        optimizer.update()

    if visualize:
        x_grid = np.mgrid[-2:3:100j, -2:3:100j]
        x1, x2 = x_grid[0], x_grid[1]
        x_grid = x_grid.reshape(2, -1).T
        y_grid = np.mean([model(x_grid).reshape(100, 100) for _ in range(100)], axis=0)

        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.ravel(), s=5)
        plt.contourf(x1, x2, y_grid, np.linspace(0, 1, 11), alpha=0.2)
        plt.colorbar()
        plt.xlim(-2, 3)
        plt.ylim(-2, 3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--visualize",
        default=0, help="plot the result if greater than 0"
    )
    args = parser.parse_args()
    main(args.visualize)
