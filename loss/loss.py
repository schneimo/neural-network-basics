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


@register('crossentropy')
class CrossEntropyLoss(LossFuncBase):

    def pred(self, pred_value):
        return softmax(pred_value)

    def calc(self, true_value, pred_value):
        num_examples = pred_value.shape[0]
        loss = (1/num_examples) * np.log(pred_value[range(num_examples), true_value])
        return -1. * np.sum(loss)

    def derivative(self, true_value, pred_value):
        num_examples = pred_value.shape[0]
        pred_value[range(num_examples), true_value] -= 1
        #pred_value /= num_examples
        return pred_value


@register('mse')
class MeanSquaredErrorLoss(LossFuncBase):

    def pred(self, pred_value):
        return pred_value

    def calc(self, true_value, pred_value):
        loss = 1/(2*pred_value.size) * np.linalg.norm(pred_value-true_value)**2
        return loss

    def derivative(self, true_value, pred_value):
        diff = pred_value-true_value
        #diff /= pred_value.shape[0]
        return diff


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
