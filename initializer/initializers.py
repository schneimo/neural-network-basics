import numpy as np


class Initializer(object):

    def create(self, shape):
        pass


class XavierInit(Initializer):

    def create(self, shape):
        n_in = shape[0]
        n_out = shape[-1]
        return np.random.randn(shape[-1], shape[0]) * np.sqrt(2.0/(n_in + n_out))
