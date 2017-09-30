import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class Model(bn.Network):

    def __init__(self):
        super().__init__(mu=np.zeros(1))

    def __call__(self, x=None):
        self.mu_prior = bn.random.Gaussian(0, 0.3, data=self.mu)
        self.x = bn.random.Gaussian(self.mu, 0.3, data=x)


def main():
    x_train = np.random.normal(loc=0.8, scale=0.1, size=2)

    sample = bn.sampler.metropolis(
        Model(),
        (x_train,),
        1000,
        10,
        mu=bn.random.Gaussian(0, 0.1)
    )
    plt.plot(
        np.linspace(-1, 1, 100),
        bn.random.Gaussian(0, 0.3).pdf(np.linspace(-1, 1, 100)).value
    )
    plt.hist(np.asarray(sample["mu"]), bins=10, alpha=0.5, normed=True)
    plt.scatter(x_train, np.random.uniform(-0.1, 0.1, size=x_train.shape))
    plt.show()


if __name__ == '__main__':
    main()
