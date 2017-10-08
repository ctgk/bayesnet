import unittest
import numpy as np
import bayesnet as bn


class TestDirichlet(unittest.TestCase):

    def test_dirichlet(self):
        np.random.seed(1234)
        obs = np.random.choice(3, 100, p=[0.2, 0.3, 0.5])
        obs = np.eye(3)[obs]
        a = bn.Parameter(np.zeros(3))
        for _ in range(100):
            a.cleargrad()
            mu = bn.softmax(a)
            d = bn.random.Dirichlet(np.ones(3) * 10, data=mu)
            x = bn.random.Categorical(mu, data=obs)
            log_posterior = x.log_pdf().sum() + d.log_pdf().sum()
            log_posterior.backward()
            a.value += 0.01 * a.grad

        count = np.sum(obs, 0) + 10
        p = count / count.sum(keepdims=True)
        self.assertTrue(np.allclose(p, mu.value, 1e-2, 1e-2))


if __name__ == '__main__':
    unittest.main()
