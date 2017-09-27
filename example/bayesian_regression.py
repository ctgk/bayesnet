import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import bayesnet as bn


class BayesianRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            mu=bn.Parameter(np.zeros((n_input, n_output))),
            logs=bn.Parameter(np.zeros((n_input, n_output)))
        )
        self.w_prior = bn.random.Gaussian(0, 10)

    def __call__(self, x, y=None):
        self.w = bn.random.Gaussian(self.mu, bn.exp(self.logs), prior=self.w_prior)
        self.y = bn.random.Gaussian(x @ self.w.draw(), 0.25, data=y)
        return self.y.mu


def main():
    degree = 9
    x_train = np.linspace(0, 1, 5)[:, None]
    y_train = np.sin(2 * np.pi * x_train) + np.random.normal(scale=0.25, size=x_train.shape)
    X_train = PolynomialFeatures(degree).fit_transform(x_train)

    model = BayesianRegressor(degree + 1, 1)
    optimizer = bn.optimizer.Adam(model, 0.01)
    for i in range(int(1e4)):
        model.cleargrad()
        loss = 0
        for _ in range(10):
            model(X_train, y_train)
            loss += model.loss() / 10
        loss.backward()
        optimizer.update()
        if i % int(1e3) == 0:
            print(loss.value)

    x = np.linspace(0, 1, 100)[:, None]
    X = PolynomialFeatures(degree).fit_transform(x)
    plt.scatter(x_train, y_train, marker="x")
    for i in range(10):
        plt.plot(x, model(X).value, color="red")
    plt.show()


if __name__ == '__main__':
    main()
