import numpy as np


# Method to implement a discrete 2D atrous convolutional operation with
def conv2d(x, filter, biases, padding, stride, dilation):
    """
    Method to implement a discrete 2D atrous convolutional operation with padding and stride

    Arguments:
        x: Input Numpy-array to transform,
        filter: Multidimensional kernel to apply
        biases: Bias vector to add to the output
        padding: Type of padding to apply before convolution
        stride: Stride steps to apply between the discrete convolutions
        dilation: Dilation rate
    """
    batch_size, in_height, in_width, in_channels = x.shape
    filter_height, filter_width, num_kernels, out_channels = filter.shape
    pad_h, pad_w = padding
    filter_height, filter_width, n_features_in, n_features_out = filter.shape

    # update effective filter shape based on dilation factor
    eff_size_height = filter_height + (filter_height - 1)(dilation - 1)
    eff_size_width = filter_width + (filter_width - 1)(dilation - 1)

    # Calculate output height and width based on the given parameters
    out_height = int((in_height + sum(pad_h) - eff_size_height) / stride + 1)
    out_width = int((in_width + sum(pad_w) - eff_size_width) / stride + 1)

    # Transform the input tensor to a vector to apply convolution as matrix multiplication
    x_vectorized = image_to_column(x, filter.shape, stride, padding, dilation)

    # Reshape kernel to
    filter_vectorized = filter.transpose(3, 2, 0, 1).reshape(out_channels, -1)

    # Apply convolution operation and reshape
    result = filter_vectorized @ x_vectorized
    result = result.reshape(out_channels, out_height, out_width, batch_size).transpose(3, 1, 2, 0)
    result += biases

    return result


# Method which calculates the padding based on the specified output shape and the
# shape of the filters
def get_padding(filter_shape, output_shape="same"):
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

    return (pad_h1, pad_h2), (pad_w1, pad_w2)


# Reference: CS231n Stanford
# https://github.com/cs231n/cs231n.github.io
def get_im2col_indices(images_shape, filter_shape, padding, stride=1, dilation=0):
    # First figure out what the size of the output should be
    batch_size, height, width, in_channels = images_shape
    filter_height, filter_width, _, _ = filter_shape

    # update effective filter shape based on dilation factor
    eff_size_height = filter_height + (filter_height - 1)(dilation - 1)
    eff_size_width = filter_width + (filter_width - 1)(dilation - 1)

    pad_h, pad_w = padding

    assert (height + np.sum(pad_h) - eff_size_height) % stride == 0
    assert (width + np.sum(pad_w) - eff_size_width) % stride == 0
    out_height = int((height + np.sum(pad_h) - eff_size_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - eff_size_width) / stride + 1)

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
    filter_height, filter_width, _, _ = filter_shape

    pad_h, pad_w = padding #get_padding(filter_shape, output_shape)

    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), pad_h, pad_w,  (0, 0)), mode='constant')

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
# TODO: Add dilation
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = get_padding(filter_shape, output_shape)  # TODO
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.empty((batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]


def softmax(x):
    exp_scores = np.exp(x)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs