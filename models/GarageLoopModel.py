import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from .base_model import Model
from layers import derived_layers


class GarageLoopModel(Model):
    def __init__(self, network_dictionary: dict):
        """
            The basic regression model used as a test bed for model construction.

        Parameters
        ----------
        network_dictionary: (dict) network configuration dictionary
        """
        super(GarageLoopModel, self).__init__(network_dictionary=network_dictionary)

        # Set the mode of the model: classifier or regression
        self.mode = 'regression'

    def build_graph(self, x: tf.Tensor) -> keras.Model:
        """
            Build the actual graph of the network.

        Parameters
        ----------
        x: (tf.Tensor) input tensor

        Returns
        -------
        model: (keras.Model) compiled Keras model ready for training
        """
        # Define the input tensor
        inputs = x

        # Perform a batch norm
        x = keras.layers.TimeDistributed(layers.BatchNormalization())(inputs)

        # Entry convolutional block
        x = derived_layers.standard_convolution_block_sequential(x, kernel_size=(7, 7),
                                                                 filters=12, dtype=self._dtype)

        # 2nd convolutional block
        x = derived_layers.standard_convolution_block_sequential(x, kernel_size=(5, 5),
                                                                 filters=24, dtype=self._dtype)

        # 3rd convolutional block
        x = derived_layers.standard_convolution_block_sequential(x, kernel_size=(3, 3),
                                                                 filters=48, dtype=self._dtype)

        # 4th grouped convolutional block
        x = derived_layers.grouped_convolution_block_sequential(x,
                                                                cardinality=4,
                                                                kernel_size=(3, 3),
                                                                channels_in=48,
                                                                batch_norm=True,
                                                                dtype=self._dtype)

        # Flatten the output
        x = layers.TimeDistributed(layers.Flatten())(x)

        # Feed it to an LSTM
        # x = layers.LSTM(units=20,
        #                 activation='relu',
        #                 recurrent_activation='hard_sigmoid',
        #                 use_bias=True,
        #                 kernel_initializer='glorot_uniform',
        #                 return_sequences=True)(x)

        # Feed it to a GRU
        x = layers.GRU(units=20,
                       activation='relu',
                       recurrent_activation='hard_sigmoid',
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       return_sequences=True)(x)

        # Output node
        x = layers.Dense(units=2, activation='linear')(x)

        # Create the keras model
        model = keras.Model(inputs=inputs, outputs=x, name='standard_regression')

        return model
