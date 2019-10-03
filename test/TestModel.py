from models import *
from layers import *
from optimizers import *
from loss import *

if __name__ == '__main__':
    model = Model(loss='crossentropylosswithlogits')
    layer1 = Dense(5, 4, activation='relu', initializer='xavier')
    model.add_layer(layer1)
    layer2 = Dense(4, 3, activation='relu', initializer='xavier')
    model.add_layer(layer2)
    optim = SGD()
    model.add_optim(optim)


    x = np.asarray([[1, 2, 3, 4, 5]])
    y = np.asarray([[1, 0, 0]])
    model.train(zip([x],[y]))
