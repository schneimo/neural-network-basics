import matplotlib.pyplot as plt

from layers import *
from models import *
from optimizers import *

if __name__ == '__main__':
    random_seed = 10
    np.random.seed(random_seed)

    # Define the network
    input_size, hidden_size, output_size = 1, 100, 1
    model = Model(loss='mse')
    layer1 = Dense(input_size, hidden_size, activation='relu', initializer='xavier', regularization=1e-3)
    model.add_layer(layer1)
    layer2 = Dense(hidden_size, output_size, activation='linear', initializer='xavier', regularization=1e-3)
    model.add_layer(layer2)
    optim = SGD(lr=0.01)
    model.add_optim(optim)

    # Training
    for i in range(100000):
        # Generate training data
        x_pts = np.random.rand(1000, 1) * 10
        y_pts = np.sin(x_pts)
        # Training the network and observing the loss
        loss, _ = model.train(x_pts, y_pts)

        if i % 10000 == 0:
            print(f"Iteration {i}: loss {loss}")

    model.set_eval()

    # Generate test data
    x_pts = np.linspace(0, 10, 1000)[:, np.newaxis]
    y_pts = np.sin(x_pts)

    # Calculate predictions
    prediction = model(x_pts)

    # Plotting true values and prediction
    x_pts, y_pts, prediction = np.squeeze(x_pts), np.squeeze(y_pts), np.squeeze(prediction)
    fig, ax = plt.subplots()
    ax.plot(x_pts, y_pts)
    ax.plot(x_pts, prediction)
    ax.set(xlabel='x', ylabel='true_val', title='sin(x)')
    ax.grid()
    plt.show()
    #fig.savefig('results/basic_regression_result.png')
