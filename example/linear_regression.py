import matplotlib.pyplot as plt
import numpy as np
import bayesnet as bn


class LinearRegressor(bn.Network):

    def __init__(self, n_input, n_output):
        super().__init__(
            w=np.zeros((n_input, n_output)),
            b=np.zeros(n_output),
            logs=1
        )

    def __call__(self, x, y=None):
        self.y = bn.random.Gaussian(x @ self.w + self.b, bn.exp(self.logs), data=y)
        return self.y.mu


def main():
    x_train = np.random.uniform(-1, 1, 10)[:, None]
    y_train = 5 * x_train - 1 + np.random.normal(scale=1., size=x_train.shape)

    model = LinearRegressor(1, 1)
    optimizer = bn.optimizer.GradientDescent(model, 0.1)
    for i in range(1000):
        model.cleargrad()
        model(x_train, y_train)
        loss = model.loss()
        loss.backward()
        optimizer.update()
        if i % 100 == 0:
            print(loss.value)

    x = np.linspace(-1, 1, 100)[:, None]
    y = model(x)
    plt.scatter(x_train, y_train, marker="x")
    plt.plot(
        x, y.value, color="orange",
        label=f"{model.w.value.item():.2}x+{model.b.value.item():.2}"
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()