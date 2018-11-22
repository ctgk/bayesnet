import argparse
import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class ARDNetwork(bn.Network):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__(
            w1_mu=np.zeros((n_input, n_hidden)),
            w1_s=np.zeros((n_input, n_hidden)),
            shape_w1=np.ones((n_input, n_hidden)),
            rate_w1=np.ones((n_input, n_hidden)),
            b1_mu=np.zeros(n_hidden),
            b1_s=np.zeros(n_hidden),
            shape_b1=np.ones(n_hidden),
            rate_b1=np.ones(n_hidden),
            w2_mu=np.zeros((n_hidden, n_output)),
            w2_s=np.zeros((n_hidden, n_output)),
            shape_w2=np.ones((n_hidden, n_output)),
            rate_w2=np.ones((n_hidden, n_output)),
            b2_mu=np.zeros(n_output),
            b2_s=np.zeros(n_output),
            shape_b2=np.ones(n_output),
            rate_b2=np.ones(n_output)
        )

    def __call__(self, x, y=None):
        ptau_w1 = bn.random.Gamma(1., 1e-2)
        self.qtau_w1 = bn.random.Gamma(bn.softplus(self.shape_w1), bn.softplus(self.rate_w1), p=ptau_w1)
        pw1 = bn.random.Gaussian(0, tau=self.qtau_w1.draw())
        self.qw1 = bn.random.Gaussian(self.w1_mu, bn.softplus(self.w1_s), p=pw1)

        ptau_b1 = bn.random.Gamma(1., 1e-2)
        self.qtau_b1 = bn.random.Gamma(bn.softplus(self.shape_b1), bn.softplus(self.rate_b1), p=ptau_b1)
        pb1 = bn.random.Gaussian(0, tau=self.qtau_b1.draw())
        self.qb1 = bn.random.Gaussian(self.b1_mu, bn.softplus(self.b1_s), p=pb1)

        ptau_w2 = bn.random.Gamma(1., 1e-2)
        self.qtau_w2 = bn.random.Gamma(bn.softplus(self.shape_w2), bn.softplus(self.rate_w2), p=ptau_w2)
        pw2 = bn.random.Gaussian(0, tau=self.qtau_w2.draw())
        self.qw2 = bn.random.Gaussian(self.w2_mu, bn.softplus(self.w2_s), p=pw2)

        ptau_b2 = bn.random.Gamma(1., 1e-2)
        self.qtau_b2 = bn.random.Gamma(bn.softplus(self.shape_b2), bn.softplus(self.rate_b2), p=ptau_b2)
        pb2 = bn.random.Gaussian(0, tau=self.qtau_b2.draw())
        self.qb2 = bn.random.Gaussian(self.b2_mu, bn.softplus(self.b2_s), p=pb2)

        h = bn.tanh(x @ self.qw1.draw() + self.qb1.draw())
        mu = h @ self.qw2.draw() + self.qb2.draw()
        self.py = bn.random.Gaussian(mu, 0.1, data=y)
        if y is None:
            return self.py.draw().value


def main(visualize):
    x_train = np.linspace(-3, 3, 10)[:, None]
    y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=x_train.shape)

    model = ARDNetwork(1, 20, 1)
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
        plt.fill_between(
            x.ravel(),
            (y_mean - y_std).ravel(), (y_mean + y_std).ravel(),
            color="orange", alpha=0.2
        )
        plt.show()

        parameter = []
        parameter.extend(list(model.w1_mu.value.ravel()))
        parameter.extend(list(model.b1_mu.value.ravel()))
        parameter.extend(list(model.w2_mu.value.ravel()))
        parameter.extend(list(model.b2_mu.value.ravel()))
        plt.plot(parameter)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--visualize",
        default=0, help="plot the result if greater than 0"
    )
    args = parser.parse_args()
    main(args.visualize)
