import pickle
from click import FileError
import numpy as np

from net.batch_iterator import BatchIterator
from net.activations import Softmax
from net.layer import AbstractLayer


class Model:
    layers = []
    loss = None
    optimizer = None
    iter = None

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except:
            print(f"[ ERROR ] Could not load from '{path}'")

        return None

    def __init__(self, layers):
        self.layers = layers
        if not isinstance(self.layers[-1], AbstractLayer):
            self.layers.append(Softmax())

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    @staticmethod
    def reshape(vec):
        return np.reshape(
            vec, newshape=(len(vec), 1, len(vec[0]) if len(vec) > 0 else 0)
        )

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def accuracy(self, x, y_true):
        y = self.forward(x)

        correct = 0
        total = len(y_true)
        for i in range(total):
            yi_true = np.where(y_true[i] == 1)[0]
            yi = np.argmax(y[i])
            correct += 1 if yi == yi_true else 0

        return correct / total * 100

    def fit(self, x, y, epochs, batch_size, test_x=None, test_y=None, log_freq=-1):
        losses = []
        acc_hist = []
        test_acc_hist = []
        # x_shaped = Model.reshape(x)
        # y_shaped = Model.reshape(y)

        self.iter = BatchIterator(x, y, batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in self.iter:
                # for i, xi in enumerate(batch_x):
                #     yi = batch_y[i]
                #     output = self.forward(xi)
                #     epoch_loss += self.loss(output, yi)

                #     error = self.loss.grad(output, yi)
                #     for layer in reversed(self.layers):
                #         error = layer.backward(error, yi)

                if hasattr(self.optimizer, "pre_train"):
                    self.optimizer.pre_train()

                output = self.forward(batch_x)
                epoch_loss = self.loss(output, batch_y)

                error = self.loss.grad(output, batch_y)
                i = len(self.layers)
                for layer in reversed(self.layers):
                    # print(i)
                    error = layer.backward(error, batch_y)
                    i -= 1

                self.optimizer()

            self.iter.reset()

            epoch_loss /= len(x)
            acc = self.accuracy(x, y)
            test_acc = 0 if test_x is None else self.accuracy(test_x, test_y)
            acc_hist.append(acc)
            test_acc_hist.append(test_acc)
            if log_freq >= 1 and (epoch + 1) % log_freq == 0:
                print(
                    f"[{epoch+1} / {epochs}] - Loss: {epoch_loss} | Acc: {acc} {f'| TestAcc: {self.accuracy(test_x, test_y)}' if test_x is not None else ''}"
                )

            losses.append(epoch_loss)

        return losses, acc_hist, test_acc_hist

    def save(self, path):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
            print(f"[ SAVED ] To '{path}'")

        except:
            print(f"[ ERROR ] Could not save to '{path}'")
