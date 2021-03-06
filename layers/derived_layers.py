import tensorflow as tf
import tensorflow.keras.initializers as initializers
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Lambda, concatenate


def standard_convolution_block(x: tf.Tensor,
                               kernel_size: (int, int),
                               filters: int,
                               batch_norm: bool = True,
                               dtype: tf.dtypes = tf.float32) -> tf.Tensor:
    """
        This is a standard sequence of deep learning components that normally accompany
        a convolution application. Note that under certain circumstances, it may be
        beneficial not to use batch normalization since the model may have some response
        latency when performing an actual drive test.

    Parameters
    ----------
    x: (tf.Tensor) input tensor
    kernel_size: (int, int) dimension of the applied kernel
    filters: (int) number of channels out
    batch_norm: (bool) are we applying batch normalization?
    dtype: (tf.dtypes) half of single precision

    Returns
    -------
    x: (tf.Tensor) output tensor
    """
    x = layers.Conv2D(kernel_size=kernel_size,
                      filters=filters,
                      padding='SAME',
                      kernel_initializer=initializers.glorot_normal,
                      use_bias=True,
                      dtype=dtype,
                      activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2),
                         padding='SAME')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x


def standard_convolution_block_sequential(x: tf.Tensor,
                                          kernel_size: (int, int),
                                          filters: int,
                                          batch_norm: bool = True,
                                          dtype: tf.dtypes = tf.float32) -> tf.Tensor:
    """
        Sister method to the above, but with the sequential 'TimeDistributed' encapsulation that
        permits for the direct incorporation of RNN/LSTM layers.

    Parameters
    ----------
    x: (tf.Tensor) input tensor
    kernel_size: (int, int) dimension of the applied kernel
    filters: (int) number of channels out
    batch_norm: (bool) are we applying batch normalization?
    dtype: (tf.dtypes) half or single precision

    Returns
    -------
    x: (tf.Tensor) output tensor
    """
    x = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                             filters=filters,
                                             padding='SAME',
                                             kernel_initializer=initializers.glorot_normal,
                                             use_bias=True,
                                             dtype=dtype,
                                             activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=(2, 2),
                                                padding='SAME'))(x)
    if batch_norm:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    return x


def grouped_convolution_block(x: tf.Tensor,
                              cardinality: int,
                              kernel_size: (int, int),
                              channels_in: int,
                              batch_norm: bool = True,
                              dtype: tf.dtypes = tf.float32) -> tf.Tensor:
    """
        This is a grouped convolution implementation where input channels are segregated
        into smaller subgroups, (i.e. the grouped channels), to which a separate bank of
        filters is applied. The implementation here does not allow for expanded output channels,
        (more channels out than in), since the authors have found that variety difficult to train,
        but would require a small alteration to the current method to implement.


    Parameters
    ----------
    x: (tf.Tensor) input tensor
    cardinality: (int) number of distinct groups to split the input channels into
    kernel_size: (int, int) dimension of applied kernel
    channels_in: (int) number of channels of the input tensor
    batch_norm: (bool) are we applying batch normalizaiton?
    dtype: (tf.dytpes) half or single precision

    Returns
    -------
    x: (tf.Tensor) output tensor
    """
    grouped_convs = []
    grouped_channels = channels_in // cardinality
    for index in range(cardinality):
        input_channels = Lambda(lambda z: z[:, :, :, index * grouped_channels:(index + 1) * grouped_channels])(x)
        group_output = layers.Conv2D(kernel_size=kernel_size,
                                     filters=grouped_channels,
                                     padding='SAME',
                                     kernel_initializer=initializers.glorot_normal,
                                     use_bias=True,
                                     dtype=dtype,
                                     activation='relu')(input_channels)
        grouped_convs.append(group_output)

    # Concatenate the output
    group_merge = concatenate(grouped_convs, axis=-1)
    x = layers.MaxPool2D(pool_size=(2, 2),
                         padding='SAME')(group_merge)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x


def grouped_convolution_block_sequential(x: tf.Tensor,
                                         cardinality: int,
                                         kernel_size: (int, int),
                                         channels_in: int,
                                         batch_norm: bool = True,
                                         dtype: tf.dtypes = tf.float32) -> tf.Tensor:
    """
        This is the sequential version of the above method, which applies a grouped convolution to an input
        tensor. This version encapsulate the key operators with 'TimeDistributed' to allow for training using
        sequential data.

    Parameters
    ----------
    x: (tf.Tensor) input tensor
    cardinality: (int) number of distinct groups to split the input channels into
    kernel_size: (int, int) dimension of applied kernel
    channels_in: (int) number of channels of the input tensor
    batch_norm: (bool) are we applying batch normalizaiton?
    dtype: (tf.dytpes) half or single precision

    Returns
    -------
    x: (tf.Tensor) output tensor
    """
    grouped_convs = []
    grouped_channels = channels_in // cardinality
    for index in range(cardinality):
        input_channels = Lambda(lambda z: z[:, :, :, index * grouped_channels:(index + 1) * grouped_channels])(x)
        group_output = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                              filters=grouped_channels,
                                              padding='SAME',
                                              kernel_initializer=initializers.glorot_normal,
                                              use_bias=True,
                                              dtype=dtype,
                                              activation='relu'))(input_channels)
        grouped_convs.append(group_output)

    # Concatenate the output
    group_merge = concatenate(grouped_convs, axis=-1)
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=(2, 2),
                               padding='SAME'))(group_merge)
    if batch_norm:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    return x
