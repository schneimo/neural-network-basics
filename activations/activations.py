import numpy as np


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
