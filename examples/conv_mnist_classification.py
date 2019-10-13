from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

from layers import *
from models import *
from optimizers import *

if __name__ == '__main__':
    # Fetch training data: MNIST
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    X = X.reshape(X.shape[0], 28, 28)
    y = y.astype(np.uint8)

    # Show the first three digits of the data
    for i in range(3):
        digit = X[i]
        plt.subplot(131+i)
        plt.axis('off')
        plt.imshow(digit)
    plt.show()

    # Split data in training and test set
    m_train = 60000
    m_test = X.shape[0] - m_train
    X_train, X_test = X[:m_train].T, X[m_train:].T
    y_train, y_test = y[:m_train].reshape(1, m_train), y[m_train:].reshape(1, m_test)

    # Define the network parameters
    hidden_size, output_size = 100, 10
    num_kernels = 32

    # Define the network layer by layer
    model = Model(loss='crossentropy')
    layer1 = Conv2D(filter_size=7, num_kernels=num_kernels, input_depth=1,
                    stride=1, dilation=0, padding='same',
                    activation='relu', initializer='xavier', regularization=1e-3)
    model.add_layer(layer1)

    layer2 = MaxPool(pool_size=4, stride=1)
    model.add_layer(layer2)

    layer3 = Flatten
    model.add_layer(layer3)

    layer4 = Dense(input_size=num_kernels, output_size=hidden_size,
                   activation='relu', initializer='xavier',
                   regularization=1e-3)
    model.add_layer(layer4)

    layer5 = Dense(input_size=hidden_size, output_size=output_size,
                   activation='linear', initializer='xavier',
                   regularization=1e-3)
    model.add_layer(layer5)

    optim = SGD(lr=1e-0)
    model.add_optim(optim)
