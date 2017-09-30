import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import bayesnet as bn


class RidgeRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            w=bn.Parameter(np.zeros((n_input, n_output)))
        )

    def __call__(self, x, y=None):
        self.w_prior = bn.random.Gaussian(0, 10, data=self.w)
        self.y = bn.random.Gaussian(x @ self.w, 0.1, data=y)
        return self.y.mu


def main():
    degree = 9
    x_train = np.linspace(0, 1, 10)[:, None]
    y_train = np.sin(2 * np.pi * x_train) + np.random.normal(scale=0.1, size=x_train.shape)
    X_train = PolynomialFeatures(degree).fit_transform(x_train)

    model = RidgeRegressor(degree + 1, 1)
    optimizer = bn.optimizer.Adam(model, 0.01)
    for i in range(int(1e5)):
        model.cleargrad()
        model(X_train, y_train)
        loss = -model.elbo()
        loss.backward()
        optimizer.update()
        if i % 1e4 == 0:
            print(loss.value)

    x = np.linspace(0, 1, 100)[:, None]
    X = PolynomialFeatures(degree).fit_transform(x)
    plt.scatter(x_train, y_train, marker="x")
    plt.plot(x, model(X).value)
    plt.show()


if __name__ == '__main__':
    main()
