import tensorflow as tf


class Model:
    def __init__(self, network_dictionary: dict):
        """
            This the base class that all defined models inherit from; at the moment
            it's pretty light on content, but as further applications are explored,
            it could be helpful to have other base methods implemented under-the-hood.


        Parameters
        ----------
        network_dictionary: (dict) configuration of network definition items
        """
        self.precision = network_dictionary['precision']

        if self.precision == 'half':
            self._dtype = tf.float16
        else:
            self._dtype = tf.float32

        # Rergression or classification mode
        self.mode = ''

    @staticmethod
    def build_graph(x: tf.Tensor) -> tf.Tensor:
        """
            Build the actual definition of the graph.

        Parameters
        ----------
        x: (tf.Tensor) input tensor to the model

        Returns
        -------
        x: (tf.Tensor) output tensor to the model
        """
        return x
