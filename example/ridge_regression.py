import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import bayesnet as bn


class LinearRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            w=bn.Parameter(np.zeros((n_input, n_output)))
        )

    def __call__(self, x):
        return x @ self.w


class RidgeRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            w=bn.Parameter(
                np.zeros((n_input, n_output)), prior=bn.random.Gaussian(0, 10.)
            )
        )

    def __call__(self, x):
        return x @ self.w


def main():
    x_train = np.linspace(0, 1, 10)[:, None]
    y_train = np.sin(2 * np.pi * x_train) + np.random.normal(scale=0.25, size=x_train.shape)
    X_train = PolynomialFeatures(degree=29).fit_transform(x_train)

    model_ml = LinearRegressor(30, 1)
    model_map = RidgeRegressor(30, 1)
    optimizer_ml = bn.optimizer.GradientDescent(model_ml.parameter, 0.01)
    optimizer_map = bn.optimizer.GradientDescent(model_map.parameter, 0.01)
    for i in range(int(1e6)):
        model_ml.cleargrad()
        model_map.cleargrad()
        y_ml = model_ml(X_train)
        y_map = model_map(X_train)
        loss_ml = bn.sum_squared_error(y_ml, y_train)
        loss_map = bn.sum_squared_error(y_map, y_train)
        loss_ml.backward()
        loss_map.backward()
        optimizer_ml.update()
        optimizer_map.update()
        if i % 1e5 == 0:
            print(loss_ml, loss_map)

    x = np.linspace(0, 1, 100)[:, None]
    X = PolynomialFeatures(9).fit_transform(x)
    plt.scatter(x_train, y_train, marker="x")
    plt.plot(x, model_ml(X).value)
    plt.plot(x, model_map(X).value)
    plt.show()


if __name__ == '__main__':
    main()
