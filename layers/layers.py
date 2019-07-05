import numpy as np
from utils import conv2D, max_pool


class Layer:

    def __init__(self):
        raise NotImplementedError

    def __call__(self, x):
        self.forward_pass(x)

    def forward_pass(self, x):
        raise NotImplementedError

    def backward_pass(self, dL_dy):
        raise NotImplementedError

    def param_gradients(self, dL_dy):
        raise NotImplementedError


class Dense(Layer):

    @property
    def z(self):
        return self.W @ self.x + self.b

    @property
    def y(self):
        return self.activation.calc(self.z)

    def dL_dx(self, dL_dy):
        dL_dz = self.activation.derivative(dL_dy)
        return self.W @ dL_dz

    def dL_dW(self, dL_dy):
        return self.activation.derivative(dL_dy).T @ self.x.T

    def dL_db(self, dL_dy):
        return self.activation.derivative(dL_dy)

    def __init__(self, shape, activation=None, initializer=None):
        self.b = np.zeros(shape[-1]).T
        self.W = initializer.create(shape)
        self.add_activation_func(activation)

    def add_activation_func(self, activation_func):
        self.activation = activation_func

    def forward_pass(self, x):
        self.x = x
        return self.y

    def backward_pass(self, dL_dy):
        return self.dL_dx(dL_dy)

    def param_gradients(self, dL_dy):
        return self.dL_dW(dL_dy), self.dL_db(dL_dy)


class Conv2D(Layer):

    def __init__(self, kernel_shape):
        self.kernel = np.random.random(kernel_shape)


class MaxPool(Layer):

    def __init__(self):
        pass


class BatchNorm(Layer):

    def __init__(self):
        pass


