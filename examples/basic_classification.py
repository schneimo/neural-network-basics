import numpy as np
import matplotlib.pyplot as plt

from layers import *
from models import *
from optimizers import *

"""
Basic example of a classification task with toy data.
Adapted from 'https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html'
"""


def generate_data(num_examples, input_dimensions, output_classes):
    X = np.zeros((num_examples*output_classes, input_dimensions))  # data matrix (each row = single example)
    y = np.zeros(num_examples*output_classes, dtype='uint8')  # class labels
    for j in range(output_classes):
        ix = range(num_examples*j, num_examples*(j+1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(num_examples)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


if __name__ == '__main__':
    random_seed = 7
    np.random.seed(random_seed)

    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes

    X, y = generate_data(N, D, K)

    # Define the network
    input_size, hidden_size, output_size = D, 100, K
    model = Model(loss='crossentropy')
    layer1 = Dense(input_size=input_size, output_size=hidden_size,
                   activation='relu', initializer='xavier', regularization=1e-3)
    model.add_layer(layer1)
    layer2 = Dense(input_size=hidden_size, output_size=output_size,
                   activation='linear', initializer='xavier', regularization=1e-3)
    model.add_layer(layer2)
    optim = SGD(lr=1e-0)
    model.add_optim(optim)

    # Training
    for i in range(10000):
        # Training the network and observing the loss
        loss = model.train(X, y)

        # Print loss every 1000 iteration
        if i % 1000 == 0:
            print("Iteration %d: loss %f" % (i, loss))

    model.set_eval()

    # evaluate training set accuracy
    prediction = model(X)
    predicted_class = np.argmax(prediction, axis=1)
    print('Training accuracy: %.2f' % (np.mean(predicted_class == y)))

    # Plotting true values and prediction
    # Creating the results and the decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    prediction = model(np.c_[xx.ravel(), yy.ravel()])
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction.reshape(xx.shape)

    # Generating the plot
    fig = plt.figure()
    plt.contourf(xx, yy, prediction, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    #fig.savefig('results/basic_classification_result.png')


