import argparse
import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class BayesianNetwork(bn.Network):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__(
            w1_mu=np.zeros((n_input, n_hidden)),
            w1_s=np.zeros((n_input, n_hidden)),
            b1_mu=np.zeros(n_hidden),
            b1_s=np.zeros(n_hidden),
            w2_mu=np.zeros((n_hidden, n_output)),
            w2_s=np.zeros((n_hidden, n_output)),
            b2_mu=np.zeros(n_output),
            b2_s=np.zeros(n_output)
        )

    def __call__(self, x, y=None):
        self.qw1 = bn.random.Gaussian(
            self.w1_mu, bn.softplus(self.w1_s),
            p=bn.random.Gaussian(0, 1)
        )
        self.qb1 = bn.random.Gaussian(
            self.b1_mu, bn.softplus(self.b1_s),
            p=bn.random.Gaussian(0, 1)
        )
        self.qw2 = bn.random.Gaussian(
            self.w2_mu, bn.softplus(self.w2_s),
            p=bn.random.Gaussian(0, 1)
        )
        self.qb2 = bn.random.Gaussian(
            self.b2_mu, bn.softplus(self.b2_s),
            p=bn.random.Gaussian(0, 1)
        )
        h = bn.tanh(x @ self.qw1.draw() + self.qb1.draw())
        mu = h @ self.qw2.draw() + self.qb2.draw()
        self.py = bn.random.Gaussian(mu, 0.1, data=y)
        if y is None:
            return self.py.draw().value


def main(visualize):
    x_train = np.linspace(-3, 3, 10)[:, None]
    y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=x_train.shape)

    model = BayesianNetwork(1, 20, 1)
    optimizer = bn.optimizer.Adam(model, 0.1)
    optimizer.set_decay(0.9, 100)

    for _ in range(10000):
        model.clear()
        model(x_train, y_train)
        elbo = model.elbo()
        elbo.backward()
        optimizer.update()

    if visualize:
        x = np.linspace(-3, 3, 1000)[:, None]
        plt.scatter(x_train, y_train)
        y = [model(x) for _ in range(100)]
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        plt.plot(x, y_mean, c="orange")
        plt.fill_between(x.ravel(), (y_mean - y_std).ravel(), (y_mean + y_std).ravel(), color="orange", alpha=0.2)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--visualize",
        default=0, help="plot the result if greater than 0"
    )
    args = parser.parse_args()
    main(args.visualize)
