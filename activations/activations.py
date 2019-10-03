import numpy as np


mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


class Activation:

    def __init__(self):
        self.y = None

    def __call__(self, z):
        return self.calc(z)

    def calc(self, z):
        raise NotImplementedError

    def derivative(self, dL_dy):
        raise NotImplementedError


@register('sigmoid')
class Sigmoid(Activation):

    def calc(self, z):
        exp = np.exp(z)
        y = self.y = 1/(1+exp)
        return y

    def derivative(self, dL_dy):
        s = self.y
        return s*(1-s)


@register('relu')
class Relu(Activation):

    def calc(self, z):
        y = self.y = np.maximum(z, 0)
        return self.y

    def derivative(self, dL_dy):
        d = self.y > 0
        return d


@register('tanh')
class Tanh(Activation):

    def calc(self, z):
        y = self.y = np.tanh(z, 0)
        return y

    def derivative(self, dL_dy):
        return 1 - self.y**2


@register('linear')
class Linear(Activation):

    def calc(self, z):
        y = self.y = z
        return y

    def derivative(self, dL_dy):
        return self.y @ dL_dy


def get_activation(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        obj = mapping[name]()
        return obj
    else:
        raise ValueError('Unknown initializers type: {}'.format(name))