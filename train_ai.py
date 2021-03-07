import numpy as np
import string
import sys
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import process_configuration
from utils.data_loader import BatchLoader
import importlib


# GPU identifier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# If multiple CUDA compatible devices are available,
# you can select an index other than 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainAI(object):
    def __init__(self, input_config: string = None):
        """
        Trains the models to be deployed to MiniAutonomous!

        Parameters
        ----------
        input_config: (string) configuration file that defines the training process

        """
        self.training_configuration = process_configuration.ConfigurationProcessor(input_config)

        # Define the model constructor
        model_definition = self._define_model()
        self.model_constructor = model_definition(self.training_configuration.network_dictionary)

        # Create the data loader
        self.data_loader = BatchLoader(self.training_configuration.data_dictionary,
                                       self.model_constructor.mode)

        # Define the image dimensions of the data
        self.image_height = self.training_configuration.data_dictionary['image_height']
        self.image_width = self.training_configuration.data_dictionary['image_width']

        # Load the data
        if self.training_configuration.data_dictionary['sequence']:
            self.training_data = self.data_loader.load_sequence_from_hdf5()
        else:
            self.training_data = self.data_loader.load_from_hdf5()

        # Define the number of samples for training
        self.n_training_samples = len(self.training_data[0][0])

    def _define_model(self) -> importlib.import_module:
        """
            Pull the model definition from the correct script in the
            models directory.

        Returns
        -------
        model: (limportlib.import_module) desired model to train
        """
        try:
            model = importlib.import_module('models.'+self.training_configuration.network_dictionary['model_name'])
            model = getattr(model, self.training_configuration.network_dictionary['model_name'])
        except ImportError:
            print('models/{}.py is not defined'.format(self.training_configuration.network_dictionary['model_name']))
        return model

    def train_model(self):
        """
            Perform actual training of the model.
        """

        # Define the input tensor dimension based on the data
        if self.training_configuration.data_dictionary['sequence']:
            sequence_length = self.training_configuration.data_dictionary['sequence_length']
            input_tensor = keras.Input(shape=(sequence_length,
                                              self.image_height,
                                              self.image_width, 3))
        else:
            input_tensor = keras.Input(shape=(self.image_height,
                                              self.image_width, 3))
        # Compile the Keras Model
        keras_model = self.model_constructor.build_graph(input_tensor)

        # Provide the user with a summary
        keras_model.summary()


if __name__ == '__main__':
    train_ai = TrainAI(sys.argv[1])

    train_ai.train_model()
