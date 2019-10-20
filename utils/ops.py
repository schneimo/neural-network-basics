import numpy as np


# Method to implement a discrete 2D atrous convolutional operation with
def conv2d(x, filter, biases, padding, stride, dilation):
    """
    Method to implement a discrete 2D atrous convolutional operation with padding and stride

    Arguments:
        x:          Input Numpy-array to transform,
        filter:     Multidimensional kernel to apply
        biases:     Bias vector to add to the output
        padding:    Type of padding to apply before convolution
        stride:     Stride steps to apply between the discrete convolutions
        dilation:   Dilation rate
    """
    batch_size, in_height, in_width, in_channels = x.shape
    filter_height, filter_width, _, out_channels = filter.shape
    pad_h, pad_w = padding

    # update effective filter shape based on dilation factor
    eff_size_height = filter_height + (filter_height - 1) * (dilation)
    eff_size_width = filter_width + (filter_width - 1) * (dilation)

    # Calculate output height and width based on the given parameters
    out_height = (in_height + sum(pad_h) - eff_size_height) // stride + 1
    out_width = (in_width + sum(pad_w) - eff_size_width) // stride + 1

    # Transform the input tensor to a vector to apply convolution as matrix multiplication
    x_reshaped = x.transpose(0, 3, 1, 2)
    x_vectorized = image_to_column(x_reshaped, filter.shape, stride, padding, dilation)

    # Reshape kernel to
    filter_vectorized = filter.transpose(3, 2, 0, 1).reshape(out_channels, -1)

    # Apply convolution operation and reshape
    result = filter_vectorized @ x_vectorized
    result = result.reshape(out_channels, out_height, out_width, batch_size).transpose(3, 1, 2, 0)
    result += biases

    return result


def pool(x, pool_func, pool_size, padding, stride):
    # TODO: Add use of pool func, to enable behaviour for average pooling!
    batch_size, in_height, in_width, in_channels = x.shape
    out_channels = in_channels
    pad_h, pad_w = padding

    # Calculate output height and width based on the given parameters
    out_height = (in_height + sum(pad_h) - pool_size) // stride + 1
    out_width = (in_width + sum(pad_w) - pool_size) // stride + 1

    # Transform the input tensor to a vector to apply convolution as matrix multiplication
    x_split = x.transpose(0, 3, 1, 2)
    x_split = x_split.reshape(batch_size * in_channels, 1, in_height, in_width)
    x_vectorized = image_to_column(x_split,
                                   (pool_size, pool_size),
                                   stride,
                                   padding,
                                   dilation=0)

    max_indexes = np.argmax(x_vectorized, axis=0)
    result = x_vectorized[max_indexes, range(max_indexes.size)]

    result = result.reshape(out_height, out_width, batch_size, out_channels)
    result = result.transpose(2, 0, 1, 3)

    return result, x_vectorized, max_indexes


# Method which calculates the padding based on the specified output shape and the
# shape of the filters
def get_padding(filter_shape, output_shape="same"):
    """
    Determines the padding for the resulting output shape given the sizes of the filter

    Arguments:
        filter_shape:   Shape of the filter. Method assumes that first two dimensions are height and width.
        output_shape:   Wanted shape of the output. Can be either 'full', 'same' or 'valid
    """

    # No padding
    if output_shape == "valid":
        pad_h1, pad_h2, pad_w1, pad_w2 = 0, 0, 0, 0

    # Output shape should be the sum of the sizes of input and kernel without stride
    elif output_shape == 'full':
        filter_height, filter_width = filter_shape[0:2]

        # output_height = (input_height + pad_h - filter_height) / stride + 1
        # Assumption: output_height = input_height + filter_height using stride = 1.
        pad_h1 = int(np.floor((2 * filter_height - 1) / 2))
        pad_h2 = int(np.ceil((2 * filter_height - 1) / 2))
        pad_w1 = int(np.floor((2 * filter_width - 1) / 2))
        pad_w2 = int(np.ceil((2 * filter_width - 1) / 2))

    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape[0:2]

        # padding = "same", this means in_height = out_height & in_width = out_width
        # pad = ((out_height - 1) * stride + filter_height - in_height) // 2

        # out_height = (in_height + pad_h - filter_height) / stride + 1
        # Assumption: out_height = in_height
        pad_h1 = int(np.floor((filter_height - 1)/2))
        pad_h2 = int(np.ceil((filter_height - 1)/2))
        pad_w1 = int(np.floor((filter_width - 1)/2))
        pad_w2 = int(np.ceil((filter_width - 1)/2))
    else:
        raise NameError(f'{output_shape} is not defined! '
                        f'Choose between same, full and valid')

    return (pad_h1, pad_h2), (pad_w1, pad_w2)


# Reference: CS231n Stanford
# https://github.com/cs231n/cs231n.github.io
def get_im2col_indices(images_shape, filter_shape, padding, stride=1, dilation=0):
    # First figure out what the size of the output should be
    batch_size, in_channels, height, width = images_shape
    filter_height, filter_width = filter_shape[0:2]

    # update effective filter shape based on dilation factor
    eff_size_height = filter_height + (filter_height - 1) * (dilation)
    eff_size_width = filter_width + (filter_width - 1) * (dilation) # dilation - 1

    pad_h, pad_w = padding

    assert (height + np.sum(pad_h) - eff_size_height) % stride == 0
    assert (width + np.sum(pad_w) - eff_size_width) % stride == 0
    out_height = (height + np.sum(pad_h) - eff_size_height) // stride + 1
    out_width = (width + np.sum(pad_w) - eff_size_width) // stride + 1

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, in_channels) * (dilation + 1)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * in_channels) * (dilation + 1)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(in_channels), filter_height * filter_width).reshape(-1, 1)

    return k, i, j


# Method which turns the image shaped input to column shape.
# Used during the forward pass of a convolution.
# Reference: CS231n Stanford
# https://github.com/cs231n/cs231n.github.io
def image_to_column(images, filter_shape, stride, padding, dilation, output_shape='same'):
    filter_height, filter_width = filter_shape[0:2]

    pad_h, pad_w = padding

    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, padding, stride, dilation)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


# Method which turns the column shaped input to image shape.
# Used during the backward pass of a convolution.
# Reference: CS231n Stanford
# https://github.com/cs231n/cs231n.github.io
def column_to_image(cols, images_shape, filter_shape, stride, padding, dilation=0, output_shape='same'):
    batch_size, height, width, channels = images_shape

    pad_h, pad_w = padding
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.empty((batch_size, channels, height_padded, width_padded), dtype=cols.dtype)

    # Calculate the indices where the dot products are applied between weights
    # and the image
    needed_im_shape = (batch_size, channels, height, width)
    k, i, j = get_im2col_indices(needed_im_shape, filter_shape,
                                 padding, stride, dilation=dilation)

    cols_reshaped = cols.reshape(channels * np.prod(filter_shape[:2]), -1, batch_size)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols_reshaped)

    # Return image without padding
    if pad_h == (0, 0) and pad_w == (0, 0):
        return images_padded
    return images_padded[:, :, pad_h[0]:-pad_h[1], pad_w[0]:-pad_w[1]]


def softmax(x):
    # TODO: Numerical stability?
    exp_scores = np.exp(x)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def incremental_mean(old, new, num_examples):
    out = old + 1/num_examples * (new - old)
    return out


def shuffle_batch(x, y, batch_size):
    assert x.shape[0] == y.shape[0]
    idx = np.random.choice(len(x), batch_size, replace=False)
    return x[idx], y[idx]


def create_minibatches(x, y, batch_size, shuffle=True):
    assert x.shape[0] == y.shape[0], \
        'Example size of input and true values are not the same'
    total = x.shape[0]

    # Shuffle the data set, by sampling random indices
    # until the size of data set is reached, if shuffle is True
    idx = np.random.choice(total, total, replace=False) if shuffle else np.arange(total)

    # Determine the step size to get arrays with #'batch_size' examples
    split = int(np.ceil(total / batch_size))

    # Create array with batches
    batches_x = np.array_split(x[idx], split, axis=0)
    batches_y = np.array_split(y[idx], split, axis=0)
    return zip(batches_x, batches_y)
