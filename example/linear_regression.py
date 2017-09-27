import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class LinearRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            w=np.zeros((n_input, n_output)),
            b=np.zeros(n_output)
        )

    def __call__(self, x):
        return x @ self.w + self.b


def main():
    x_train = np.random.uniform(-1, 1, 10)[:, None]
    y_train = 5 * x_train - 1 + np.random.normal(scale=1., size=x_train.shape)

    model = LinearRegressor(1, 1)
    optimizer = bn.optimizer.GradientDescent(model.parameter, 0.1)
    for _ in range(1000):
        model.cleargrad()
        y = model(x_train)
        loss = bn.mean_squared_error(y, y_train)
        loss.backward()
        optimizer.update()

    x = np.linspace(-1, 1, 100)[:, None]
    y = model(x).value
    plt.scatter(x_train, y_train, marker="x")
    plt.plot(
        x, y, color="orange",
        label=f"{model.w.value.item():.2}x+{model.b.value.item():.2}"
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()