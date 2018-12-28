import argparse
import numpy as np
import scipy.stats as st
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import bayesnet as bn


class CNN(bn.Network):

    def __init__(self):
        super().__init__()
        truncnorm = st.truncnorm(a=-2, b=2, scale=0.1)
        with self.set_parameter():
            self.w1 = truncnorm.rvs((5, 5, 1, 20))
            self.b1 = np.zeros(20) + 0.1
            self.w2 = truncnorm.rvs((5, 5, 20, 20))
            self.b2 = np.zeros(20) + 0.1
            self.w3 = truncnorm.rvs((4 * 4 * 20, 500))
            self.b3 = np.zeros(500) + 0.1
            self.w4 = truncnorm.rvs((500, 10))
            self.b4 = np.zeros(10) + 0.1

    def __call__(self, x, y=None):
        h = bn.relu(bn.convolve2d(x, self.w1) + self.b1)
        h = bn.max_pooling2d(h, (2, 2), (2, 2))

        h = bn.relu(bn.convolve2d(h, self.w2) + self.b2)
        h = bn.max_pooling2d(h, (2, 2), (2, 2))

        h = h.reshape(-1, 4 * 4 * 20)
        h = bn.relu(h @ self.w3 + self.b3)
        h = bn.dropout(h, 0.2, is_dropping=(y is not None))

        self.py = bn.random.Categorical(logit=h @ self.w4 + self.b4, data=y)
        return self.py.mu.value


def main(args):
    np.random.seed(1234)

    mnist = fetch_mldata("MNIST original")
    x = mnist.data
    label = mnist.target
    x = x / np.max(x, axis=1, keepdims=True)
    x = x.reshape(-1, 28, 28, 1)

    x_train, x_test, label_train, label_test = train_test_split(x, label, test_size=0.1)
    y_train = LabelBinarizer().fit_transform(label_train)

    cnn = CNN()
    optimizer = bn.optimizer.Adam(cnn, 1e-3)

    bn.Config.is_training = True
    while True:
        indices = np.random.permutation(len(x_train))
        for index in range(0, len(x_train), args.minibatch):
            cnn.clear()
            x_batch = x_train[indices[index: index+args.minibatch]]
            y_batch = y_train[indices[index: index+args.minibatch]]
            proba = cnn(x_batch, y_batch)
            log_likelihood = cnn.log_pdf()
            if optimizer.n_iter % (args.iteration // 100) == 0:
                accuracy = accuracy_score(np.argmax(y_batch, axis=-1), np.argmax(proba, axis=-1))
                print(f"step {optimizer.n_iter:04}", end=", ")
                print(f"accuracy {accuracy:.2f}", end=", ")
                print(f"Log Likelihood {log_likelihood.value:g}")
            log_likelihood.backward()
            optimizer.update()
            if optimizer.n_iter == args.iteration:
                break
        else:
            continue
        break

    bn.Config.is_training = False
    label_pred = []
    for i in range(0, len(x_test), 50):
        label_pred.append(np.argmax(cnn(x_test[i: i+50]), axis=-1))
    label_pred = np.asarray(label_pred).ravel()
    print("accuracy: ", accuracy_score(label_test, label_pred))
    print("Confusion matrix: (truth x prediction)")
    print(confusion_matrix(label_test, label_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iteration", type=int, default=10000, help="number of iterations to update parameter")
    parser.add_argument("-m", "--minibatch", type=int, default=10, help="minibatch size for training")
    args = parser.parse_args()
    main(args)
