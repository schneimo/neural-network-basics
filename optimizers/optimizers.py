import numpy as np


class Optimizer:

    def __init__(self, lr):
        self.lr = lr
        self.cache = {}
        self.cur_step = 0

    def step(self):
        """Increment step counter"""
        self.cur_step += 1

    def __call__(self, grad, param):
        return self.update(grad, param)

    def update(self, grad, param):
        return NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.001, momentum=0):
        super().__init__(lr)
        self.momentum = momentum

    def update(self, grad, param):
        self.step()
        #avg_grad = grad.mean(axis=1)[:, np.newaxis]

        param_id = id(param)

        if param_id not in self.cache:
            self.cache[param_id] = np.zeros_like(grad)

        update = self.cache[param_id] = self.momentum * self.cache[param_id] + self.lr * grad
        return update


class AdaGrad(Optimizer):
    pass


class RMSProp(Optimizer):
    pass


class Adam(Optimizer):
    pass

