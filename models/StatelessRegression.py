import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from .base_model import Model
from custom_layers import derived_layers


class StatelessRegression(Model):
    def __init__(self, network_dictionary: dict):
        """
            The basic regression model used as a test bed for model construction.

        Parameters
        ----------
        network_dictionary: (dict) network configuration dictionary
        """
        super(StatelessRegression, self).__init__(network_dictionary=network_dictionary)

        # Set the mode of the model: classifier or regression
        self.mode = 'regression'

    def build_graph(self, x: tf.Tensor) -> keras.Model:
        """
            Build the actual graph (this time using Keras)

        Parameters
        ----------
        x : (tf.Tensor) input tensor

        Returns
        -------
        model: (keras.Model) compiled Keras model ready for training
        """
        # Define the input tensor
        inputs = x

        # Perform batch norm on the input tensor
        x = layers.BatchNormalization()(inputs)

        # Entry convolutional block
        x = derived_layers.standard_convolution_block(x, kernel_size=(7, 7), filters=16, dtype=self._dtype)

        # 2nd convolution block
        x = derived_layers.standard_convolution_block(x, kernel_size=(7, 7), filters=16, dtype=self._dtype)

        # 3rd convolution block
        x = derived_layers.standard_convolution_block(x, kernel_size=(5, 5), filters=64, dtype=self._dtype)

        # 4th convolution block
        x = derived_layers.standard_convolution_block(x, kernel_size=(3, 3), filters=64, dtype=self._dtype)

        # Flatten the output
        x = layers.Flatten()(x)

        # Fully-connected layer
        x = layers.Dense(units=60,
                         activation='relu',
                         kernel_initializer=tf.initializers.glorot_normal(),
                         name='dense_large')(x)

        # Output
        x = layers.Dense(units=2,
                         activation='linear',
                         kernel_initializer=tf.initializers.glorot_normal(),
                         name='dense')(x)

        # Create the keras model
        model = keras.Model(inputs=inputs, outputs=x, name='simple_regression')

        return model
