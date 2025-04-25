from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]



class MomentGD(Optimizer):
    def __init__(self, init_lr, model, beta):
        super().__init__(init_lr, model)
        self.beta = beta
        self.prev_params = {}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                if layer not in self.prev_params:
                    # 冷启动
                    self.prev_params[layer] = {
                        key: np.copy(layer.params[key]) for key in layer.params.keys()
                    }

                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    if layer.grads[key] is not None:
                        w_t = layer.params[key]
                        w_prev = self.prev_params[layer][key]
                        layer.params[key] += -self.init_lr * layer.grads[key] + self.beta * (w_t - w_prev)
                        self.prev_params[layer][key] = np.copy(w_t)
