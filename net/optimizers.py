from .layer import EPSILON, Dense
import numpy as np


class Optimizer:
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self):
        pass


class GradientDescent(Optimizer):
    def __init__(self, model, learning_rate):
        super().__init__(model)
        self.learning_rate = learning_rate

    def __call__(self):
        lr = self.learning_rate
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weights -= layer.weights_grad * lr
                layer.bias -= layer.bias_grad * lr


class CachedOptimizer(Optimizer):
    def init_cache(self, keys):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                for key, shape_key in keys.items():
                    layer.cache[key] = np.zeros(shape=layer.__dict__[shape_key].shape)

    def update_cache(self):
        pass

    def __call__(self):
        self.update_cache()


class Momentum(CachedOptimizer):
    def __init__(self, model, learning_rate, beta, **kwargs):
        super().__init__(model=model, learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate
        self.beta = beta
        self.init_cache(
            {
                "weights_m": "weights",
                "bias_m": "bias",
            }
        )

    def update_cache(self):
        lr = self.learning_rate
        beta = self.beta
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.cache["weights_m"] = (
                    layer.cache["weights_m"] * beta - layer.weights_grad * lr
                )
                layer.cache["bias_m"] = (
                    layer.cache["bias_m"] * beta - layer.bias_grad * lr
                )

    def __call__(self):
        super().__call__()

        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weights += layer.cache["weights_m"]
                layer.bias += layer.cache["bias_m"]


class NAG(CachedOptimizer):
    def __init__(self, model, learning_rate, beta, **kwargs):
        super().__init__(model=model, learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate
        self.beta = beta
        self.init_cache(
            {
                "weights_mn": "weights",
                "bias_mn": "bias",
            }
        )

    def update_cache(self):
        lr = self.learning_rate
        beta = self.beta
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.cache["weights_mn"] = (
                    layer.cache["weights_mn"] * beta - layer.weights_grad * lr
                )
                layer.cache["bias_mn"] = (
                    layer.cache["bias_mn"] * beta - layer.bias_grad * lr
                )

    def __call__(self):
        super().__call__()
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weights += layer.cache["weights_mn"]
                layer.bias += layer.cache["bias_mn"]

    def pre_train(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weights += layer.cache["weights_mn"]
                layer.bias += layer.cache["bias_mn"]


class AdaGrad(CachedOptimizer):
    def __init__(self, model, learning_rate):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.init_cache(
            {
                "weights_g": "weights",
                "bias_g": "bias",
            }
        )

    def update_cache(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.cache["weights_g"] += layer.weights_grad**2
                layer.cache["bias_g"] += layer.bias_grad**2

    def __call__(self):
        super().__call__()

        lr = self.learning_rate
        for layer in self.model.layers:
            if isinstance(layer, Dense):

                layer.weights -= (
                    lr
                    / ((layer.cache["weights_g"] + EPSILON) ** 0.5)
                    * layer.weights_grad
                )
                layer.bias -= (
                    lr / ((layer.cache["bias_g"] + EPSILON) ** 0.5) * layer.bias_grad
                )


class RMSProp(CachedOptimizer):
    def __init__(self, model, learning_rate, lam, **kwargs):
        super().__init__(model=model, learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate
        self.lam = lam
        self.init_cache(
            {
                "weights_e": "weights",
                "bias_e": "bias",
            }
        )

    def update_cache(self):
        lam = self.lam
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.cache["weights_e"] = (
                    lam * layer.cache["weights_e"] + (1 - lam) * layer.weights_grad**2
                )
                layer.cache["bias_e"] = (
                    lam * layer.cache["bias_e"] + (1 - lam) * layer.bias_grad**2
                )

    def __call__(self):
        super().__call__()
        lr = self.learning_rate
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weights -= (
                    lr
                    / ((layer.cache["weights_e"] + EPSILON) ** 0.5)
                    * layer.weights_grad
                )
                layer.bias -= (
                    lr / ((layer.cache["bias_e"] + EPSILON) ** 0.5) * layer.bias_grad
                )


class Adam(RMSProp, Momentum):
    def __init__(self, model, learning_rate, beta, lam):
        super().__init__(model=model, learning_rate=learning_rate, lam=lam, beta=beta)

    def update_cache(self):
        RMSProp.update_cache(self)
        Momentum.update_cache(self)

    def __call__(self):
        CachedOptimizer.__call__(self)
        lr = self.learning_rate

        for layer in self.model.layers:
            if isinstance(layer, Dense):

                layer.weights += (
                    lr
                    / ((layer.cache["weights_e"] + EPSILON) ** 0.5)
                    * layer.cache["weights_m"]
                )
                layer.bias += (
                    lr
                    / ((layer.cache["bias_e"] + EPSILON) ** 0.5)
                    * layer.cache["bias_m"]
                )
