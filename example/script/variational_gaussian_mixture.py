import argparse
import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class VariationalGaussianMixture(bn.Network):

    def __init__(self, n_component):
        self.n_component = n_component
        super().__init__(
            c=np.ones(n_component),
            m=np.linspace(-10, 10, n_component),
            s=np.ones(n_component),
            shape=np.ones(n_component),
            rate=np.ones(n_component)
        )

    def gaussian(self, x, z):
        self.qtau = bn.random.Gamma(self.shape, self.rate, p=bn.random.Gamma(1., 1.))
        tau = self.qtau.draw()
        pmu = bn.random.Gaussian(0., tau=tau)
        self.qmu = bn.random.Gaussian(self.m, bn.softplus(self.s), p=pmu)
        self.px = bn.random.GaussianMixture(z, self.qmu.draw(), 1 / bn.sqrt(tau), data=x)

    def category(self, z):
        coef = bn.softmax(self.c)
        self.pc = bn.random.Dirichlet(np.ones(coef.shape) * 1., data=coef)
        self.pz = bn.random.Categorical(coef, data=z)

    def __call__(self, x, z=None):
        if z is None:
            self.gaussian(x, bn.softmax(self.c))
            return self.px.pdf().value
        self.category(z)
        self.gaussian(x, z)
        return self.pz.pdf().value * self.px.pdf().value


def main(visualize):
    x_train = np.array([
        np.random.normal(loc=-7.5, scale=1, size=100),
        np.random.normal(loc=-2.5, scale=1, size=100),
        np.random.normal(loc=5, scale=2, size=100)
    ]).flatten()

    model = VariationalGaussianMixture(3)
    optimizer_c = bn.optimizer.Adam([model.c], 1e-3)
    optimizer_g = bn.optimizer.Adam([model.m, model.s, model.shape, model.rate], 1e-3)

    for _ in range(10):
        resp = 0
        for _ in range(10):
            resp_ = np.stack(
                [model(x_train[:, None], np.eye(model.n_component)[i]) for i in range(model.n_component)],
                axis=-1
            )
            resp_ /= resp_.sum(axis=-1, keepdims=True)
            resp += resp_ / 10
        for _ in range(100):
            model.clear()
            model.category(resp)
            log_posterior = model.log_pdf()
            log_posterior.backward()
            optimizer_c.update()
        for _ in range(1000):
            model.clear()
            model.gaussian(x_train[:, None], resp)
            elbo = model.elbo()
            elbo.backward()
            optimizer_g.update()

    if visualize:
        resp = 0
        for _ in range(10):
            resp_ = np.stack(
                [model(x_train[:, None], np.eye(model.n_component)[i]) for i in range(model.n_component)],
                axis=-1
            )
            resp_ /= resp_.sum(axis=-1, keepdims=True)
            resp = resp + resp_ / 10
        plt.scatter(x_train, np.random.normal(scale=0.005, size=x_train.size), s=5, c=resp)
        plt.hist(x_train, bins=20, density=True, alpha=0.2)

        x = np.linspace(-10, 10, 1000)
        p = np.mean([model(x[:, None]) for _ in range(1000)], axis=0)
        plt.plot(x, p)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--visualize",
        default=0, help="plot the result if greater than 0"
    )
    args = parser.parse_args()
    main(args.visualize)