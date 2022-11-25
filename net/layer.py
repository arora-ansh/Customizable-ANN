from distutils.log import debug
import numpy as np

EPSILON = 1e-12


class AbstractLayer:
    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Dense(AbstractLayer):

    weights = None
    bias = None
    weights_grad = None
    bias_grad = None
    x = None
    debug = None
    cache = None

    def __init__(self, input_size, output_size, debug=False):
        self.weights = (np.random.rand(input_size, output_size) - 0.5) * 1e-1
        self.bias = (np.random.rand(1, output_size) - 0.5) * 1e-1
        self.weights_grad = np.zeros(shape=(input_size, output_size))
        self.bias_grad = np.zeros(shape=(1, output_size))
        self.debug = debug
        self.cache = {}

    def clear_grads(self):
        self.weights_grad.fill(0)
        self.bias_grad.fill(0)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad, *args):
        self.bias_grad = np.mean(grad, axis=0)
        self.weights_grad = (self.x.T @ grad) / self.x.shape[0]
        return grad @ self.weights.T
