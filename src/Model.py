import numpy as np


class Model:

    def __init__(self):
        self.layers = []

    def add_loss(self, loss_func):
        self.loss_func = loss_func

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_initializer(self, initializer):
        self.initializer = initializer

    def add_learning_rate(self, epsilon):
        self.epsilon = epsilon

    def forward_pass(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output

    def predict(self, input):
        return self.forward_pass(input)

    def backward_pass(self, y):
        self.loss_func.calc(y)
        dL_dy = self.loss_func.derivative(y)
        for layer in self.layers:
            dL_dW, dL_db = layer.param_gradients(dL_dy)
            layer.W -= self.epsilon * dL_dW
            layer.b -= self.epsilon * dL_db
            dL_dy = layer.backward_pass(dL_dy)

    def train(self, data):
        for x, y in data:
            self.forward_pass(x)
            self.backward_pass(y)


class Layer:

    def __init__(self):
        raise NotImplementedError

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

    def __init__(self, shape, activation, initializer):
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


class Initializer(object):

    def create(self, shape):
        pass


class XavierInit(Initializer):

    def create(self, shape):
        n_in = shape[0]
        n_out = shape[-1]
        return np.random.randn(shape[-1], shape[0]) * np.sqrt(2.0/(n_in + n_out))


class Conv(Layer):

    def __init__(self):
        pass


class MaxPool(Layer):

    def __init__(self):
        pass


class Activation:

    def calc(self, z):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError


class Softmax(Activation):

    @property
    def y(self):
        exp_scores = np.exp(self.z)
        return exp_scores / np.sum(exp_scores)

    def calc(self, z):
        self.z = z
        return self.y

    def derivative(self, dL_dy):
        s = np.matmul(dL_dy, self.y)
        return s - np.sum(s)


class Relu(Activation):

    @property
    def y(self):
        return np.maximum(self.z, 0)

    def calc(self, z):
        self.z = z
        return self.y

    def derivative(self, dL_dy):
        d = np.where(self.y > 0, 1, 0)
        return d @ dL_dy.T


class CrossEntropyLossWithLogits:

    def __init__(self, softmax):
        self.softmax = softmax

    def calc(self, p):
        self.y = p * np.log(self.softmax.y)
        return -1 * np.sum(self.y)

    def derivative(self, p):
        return self.y - p


class MeanSquaredErrorLoss:

    def calc(self, y):
        return



