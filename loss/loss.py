import numpy as np
from utils import softmax

mapping = {}
def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


class LossFuncBase:

    def __call__(self, true_value, pred_value):
        return self.calc(true_value, pred_value)

    def calc(self, true_value, pred_value):
        return NotImplementedError


@register('crossentropylosswithlogits')
class CrossEntropyLossWithLogits(LossFuncBase):

    def pred(self, pred_value):
        return softmax(pred_value)

    def calc(self, true_value, pred_value):
        self.y = true_value * np.log(softmax(pred_value))
        return -1 * np.sum(self.y)

    def derivative(self, true_value):
        return self.y - true_value


@register('mse')
class MeanSquaredErrorLoss(LossFuncBase):

    def pred(self, pred_value):
        return pred_value

    def calc(self, true_value, pred_value):
        self.y = y = 0.5*np.mean(np.square(true_value-pred_value))
        return y

    def derivative(self, true_value):
        return self.y - true_value


def get_loss(name):
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
