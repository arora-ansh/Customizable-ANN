from tkinter import Y
import numpy as np
from net.layer import EPSILON, AbstractLayer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    x = np.exp(x - np.max(x))
    res = x / (x.sum() + EPSILON)
    return res


def sigmoid_back(x, grad, y):
    val = sigmoid(x)
    return (val * (1.0 - val)) * grad


def relu_back(x, grad, y):
    res = (x > 0) * grad
    return res


def tanh_back(x, grad, y):
    val = tanh(x)
    return (1.0 - (val**2)) * grad


def softmax_back(x, grad, y):
    return grad


class Activation(AbstractLayer):

    x = None

    def __init__(self, fn, fn_back):
        self.fn = fn
        self.fn_back = fn_back

    def forward(self, x):
        self.x = x
        return self.fn(x)

    def backward(self, grad, y):
        # print(grad)

        return self.fn_back(self.x, grad, y)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_back)


class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_back)


class TanH(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_back)


class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_back)
