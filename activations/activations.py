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

    def derivative(self):
        raise NotImplementedError


@register('sigmoid')
class Sigmoid(Activation):

    def calc(self, z):
        exp = np.exp(z)
        y = self.y = 1. / (1. + exp)
        return y

    def derivative(self):
        s = self.y
        return s * (1. - s)


@register('relu')
class Relu(Activation):

    def calc(self, z):
        y = self.y = z * (z > 0)
        return y

    def derivative(self):
        d = 1. * (self.y > 0)
        return d


@register('tanh')
class Tanh(Activation):

    def calc(self, z):
        y = self.y = np.tanh(z)
        return y

    def derivative(self):
        return 1. - self.y**2


@register('linear')
class Linear(Activation):

    def calc(self, z):
        y = self.y = z
        return y

    def derivative(self):
        return np.ones_like(self.y)


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