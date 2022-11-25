import numpy as np

from net.layer import EPSILON


def mse(y, y_true):
    return np.sum(np.power(y_true - y, 2))


def mse_grad(y, y_true):
    res = 2 * (y - y_true)
    return res


def cross_entropy(y, y_true):
    loss = -np.sum(y_true * np.log(y + EPSILON))
    return loss


def cross_entropy_grad(y, y_true):
    return y - y_true


class Loss:
    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad

    def __call__(self, y, y_true):
        return self.fn(y, y_true)

    def grad(self, y, y_true):
        return self.fn_grad(y, y_true)


class MSE(Loss):
    def __init__(self):
        super().__init__(mse, mse_grad)


class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__(cross_entropy, cross_entropy_grad)
