import numpy as np

mapping = {}
def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


class Initializer(object):

    def __call__(self, shape):
        return NotImplementedError


@register('xavier')
class Xavier(Initializer):

    def __call__(self, shape):
        fan_in, fan_out = shape
        b = np.sqrt(6/(fan_in + fan_out))
        return np.random.uniform(-b, b, size=shape)


@register('he')
class He(Initializer):

    def __call__(self, shape):
        fan_in, _ = shape
        b = np.sqrt(6/fan_in)
        return np.random.uniform(-b, b, size=shape)


def get_initializer(name):
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
        return mapping[name]()
    else:
        raise ValueError('Unknown initializers type: {}'.format(name))
