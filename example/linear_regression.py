import matplotlib.pyplot as plt
import numpy as np
import bayes


def model(x, w, s):
    return bayes.random.Gaussian(mu=x * w, sigma=bayes.function.exp(s))

def main():
    x = np.linspace(0, 1, 100)
    t = x * 5 + np.random.normal(scale=1., size=100)

    w = bayes.tensor.Parameter(0.)
    s = bayes.tensor.Parameter(0.)
    optimizer = bayes.optimizer.GradientDescent([w, s], 0.01)
    for _ in range(1000):
        optimizer.cleargrad()
        y = model(x, w, s)
        loss = y.nll(t)
        loss.backward()
        optimizer.update()

    y = model(x, w, s)
    plt.scatter(x, t)
    plt.plot(x, y.mean, color="orange", label=f"w={w.value}")
    plt.fill_between(
        x, y.mean - y.var ** 0.5, y.mean + y.var ** 0.5,
        alpha=0.5, label=f"std={y.var ** 0.5}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
