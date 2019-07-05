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

