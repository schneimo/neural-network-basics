import numpy as np


def conv2D(input, weights, biases, stride=1, padding='SAME'):
    """Convolutional layer with filter size 3x3 and 'same' padding.
        `x` is a NumPy array of shape [height, width, n_features_in]
        `weights` has shape [3, 3, n_features_in, n_features_out]
        `biases` has shape [n_features_out]
        Return the output of the 3x3 conv (without activation)
        """

    """
    This is an implementation which should be work similar like TensorFlow. As far as I know it uses
    a Toeplitz matrix to perform the convolution as a dot product, what is also called im2col.
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/conv2d
    """
    in_height, in_width, channels = input.shape
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


def max_pool(input, pool_size=2, stride=2):
    """Max pooling with pool size 2x2 and stride 2.
    `input` is a numpy array of shape [height, width, n_features]
    """
    input_height, input_width, channels = input.shape

    output_height = (input_height - pool_size)//stride + 1
    output_width = (input_width - pool_size)//stride + 1

    result = np.zeros((output_height, output_width, channels))
    for c in range(channels):
        for i in range(output_height):
            for j in range(output_width):
                k, l = i*stride, j*stride
                result[i, j, c] = input[k: (k + pool_size), l: (l + pool_size), c].max()

    return result
