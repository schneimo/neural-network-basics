import numpy as np
from initializers import get_initializer
from activations import get_activation
from utils import image_to_column, column_to_image, determine_padding


class Layer:

    def __init__(self, activation, initializer, regularization=0.0):
        self.train = True
        self._optimizer = None
        self.reg = regularization
        self.add_activation(activation)
        self.add_initializer(initializer)

    def __call__(self, x):
        return self.forward_pass(x)

    def forward_pass(self, x):
        raise NotImplementedError

    def backward_pass(self, dL_dy):
        raise NotImplementedError

    def gradients(self, dL_dy):
        raise NotImplementedError

    def set_train(self):
        self.train = True
    
    def set_eval(self):
        self.train = False

    def add_initializer(self, initializer):
        self._initializer = get_initializer(initializer)

    def add_activation(self, activation_func):
        self._activation = get_activation(activation_func)
    
    def add_optim(self, optimizer):
        self._optimizer = optimizer


class Flatten(Layer):
    """
    Takes a multidimensional Numpy-Array as input and returns a vectorized form of it
    to pass it to a fully connected/dense layer

    """
    pass


class Dense(Layer):
    """
    A regular neural network layer, this means a layer in which every unit is 
    connected to every unit of the last layer.

    The output is computed as follow:
        z = x dot W + b
        y = activation(z)

        x should have shape of (batch_size, #inputs == #units)
        y should have shape of (batch_size, #outputs)


    Arguments:
        input_size: Length of the flattend input vector as integer
        output_size: Number of units of this layer, which is the same of the output size
        activation: Activation function to use in the computation
        initializer: Weight initializers to set the values of the weights and biases
    """

    def __repr__(self):
        return f"Dense Layer {self.W.shape}"

    def __init__(self, input_size, output_size, activation='linear', initializer='glorot', regularization=0.0):
        super().__init__(activation=activation, initializer=initializer, regularization=regularization)
        self.b = self._initializer((1, output_size))  # weight shape: (output_size, )
        self.W = self._initializer((input_size, output_size))  # weight shape: (input_size, output_size)
        self.X = None  # Placeholder to save the input batch tensor

    def forward_pass(self, x):
        self.X = x
        z = self.X @ self.W + self.b
        y = self._activation.calc(z=z)
        return y

    def backward_pass(self, dL_dy):
        """
        Arguments:
            dL_dy: Gradient tensor of the next layer
            lr: Learning rate to use to update weights and biases
        """
        assert self.train, 'Layer not set to train!'
        assert self.X is not None, 'No forward pass before!'

        dx, dw, db = self.gradients(dL_dy)

        dw += self.reg * self.W
        db += self.reg * self.b

        self.W -= self._optimizer(dw, self.W)
        self.b -= self._optimizer(db, self.b)

        return dx

    def gradients(self, dL_dy):
        dL_dz = dL_dy * self._activation.derivative()
        dL_dx = dL_dz @ self.W.T
        dL_dw = self.X.T @ dL_dz
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        return dL_dx, dL_dw, dL_db


class Conv2D(Layer):

    def __init__(self, num_kernels, kernel_shape, stride=1, padding=None, activation='linear', initializer='glorot'):
        if initializer is None:
            initializer = None # TODO
        self.b = np.zeros(num_kernels)
        self.W = initializer.create(kernel_shape)
        self.kernel = np.random.random(kernel_shape)
    
    def forward_pass(self, x):
        """
        Convolutional layer with filter size 3x3 and 'same' padding.
        `x` is a NumPy array of shape [height, width, n_features_in]
        `weights` has shape [3, 3, n_features_in, n_features_out]
        `biases` has shape [n_features_out]
        Return the output of the 3x3 conv (without activation)

        This is an implementation which should be work similar like TensorFlow. As far as I know it uses
        a Toeplitz matrix to perform the convolution as a dot product, what is also called im2col.
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/conv2d
        """

        in_height, in_width, channels = x.shape
        filter_height, filter_width, n_features_in, n_features_out = weights.shape

        out_height = in_height
        out_width = in_width

        # padding = "same", this means in_height = out_height & in_width = out_width
        pad = ((out_height - 1) * stride + filter_height - in_height) // 2

        paddings = np.array([[pad, pad], [pad, pad], [0, 0]])
        x_padded = np.pad(input, paddings, 'constant')

        """
        From Tensorflow 2 doc: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/conv2d
        Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor 
        of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following:

        1. Flattens the filter to a 2-D matrix 
        with shape [filter_height * filter_width * in_channels, output_channels].

        2. Extracts image patches from the input tensor to form a virtual tensor of 
        shape [batch, out_height, out_width, filter_height * filter_width * in_channels].

        3. For each patch, right-multiplies the filter matrix and the image patch vector.
        """
        # In other words this means:
        # Calculate and use Toplitz-matrix to perform the convolution as simple dot-product like TF does (im2col).
        # The convolution/weight tensor gets flattend and from the padded input tensor, we extract so called image patches.
        # Image patches means, that we take the regions where convolution is performed in the for-loop otherwise and stack
        # them together into a large vector. To be able to perform a simple dot-product we have to reshape this vector
        # into a matrix.

        weights_flat = np.reshape(weights, [filter_height * filter_width * n_features_in, n_features_out])

        windows = []
        for y in range(out_height):
            for x in range(out_width):
                k, l = y * stride, x * stride
                window = x_padded[k: k + filter_height, l: l + filter_width, :]
                windows.append(window)
        stacked = np.stack(windows)
        x_patched = np.reshape(stacked, [-1, n_features_in * filter_width * filter_height])

        result = np.matmul(x_patched, weights_flat) + biases
        result = np.reshape(result, [out_height, out_width, n_features_out])
        
        return result


class MaxPool(Layer):

    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size
        pass

    def forward_pass(self, x):
        """Max pooling with pool size 2x2 and stride 2.
        `input` is a numpy array of shape [height, width, n_features]
        """
        input_height, input_width, channels = x.shape

        output_height = (input_height - self.pool_size)//self.stride + 1
        output_width = (input_width - self.pool_size)//self.stride + 1

        result = np.zeros((output_height, output_width, channels))
        for c in range(channels):
            for i in range(output_height):
                for j in range(output_width):
                    k, l = i*stride, j*stride
                    result[i, j, c] = input[k: (k + self.pool_size), l: (l + self.pool_size), c].max()

        return result


class AveragePool(Layer):

    def __init__(self):
        pass


class BatchNorm2D(Layer):

    def __init__(self):
        pass


class BatchNorm3D(Layer):

    def __init__(self):
        pass


