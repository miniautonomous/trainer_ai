import numpy as np
import string
import sys
import os
import tensorflow as tf
from tensorflow import keras
from utils import process_configuration
from utils.data_loader import BatchLoader
from utils.plot_results import plot_results
import importlib
from tensorflow.keras import backend as K


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
        self.n_validation_samples = len(self.training_data[1][0])

    def create_dataset(self, images: np.array, labels: np.ndarray) -> tf.data.Dataset:
        """
            Create a Tensorflow Dataset based on images and corresponding labels.

        Parameters
        ----------
        images: (np.ndarray) image data set
        labels: (np.ndarray) corresponding labels

        Returns
        -------
        tf_dataset: (tf.data.Dataset) tf dataset comprised of the images and labels
        """
        if not self.training_configuration.data_dictionary['large_data']:
            tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))\
                .batch(batch_size=self.training_configuration.training_dictionary['batch_size'],
                       drop_remainder=True)\
                .cache()\
                .repeat()
        else:
            image_blocks = np.array_split((images, 2))
            label_blocks = np.array_split((labels, 2))
            dataset_a = tf.data.Dataset.from_tensor_slices((image_blocks[0], label_blocks[0]))
            dataset_b = tf.data.Dataset.from_tensor_slices((image_blocks[1], label_blocks[1]))
            tf_dataset = dataset_a.concatenate(dataset_b)\
                .batch(batch_size=self.training_configuration.training_dictionary['batch_size'],
                       drop_remainder=True)\
                .cache().\
                repeat()

        return tf_dataset

    @staticmethod
    def r2_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor):
        """
            Callculate the R2 value for regression models.

        Parameters
        ----------
        y_true: (tf.Tensor) actual value from data
        y_pred: (tf.Tensor) inference output from model

        Returns
        -------
        R2: (float) R2 metric
        """
        residual = K.sum(K.square(y_true - y_pred))
        total = K.sum(K.square(y_true - K.mean(y_true)))
        r2 = 1 - residual / (total + K.epsilon())
        return r2

    def define_loss_and_metric(self):
        """
            Defines the loss (and metric( according to the configuration scripts

        Returns
        -------
        loss_type: (tf.keras.losses) type of loss desired for simulation
        metric: (string) type of metric to use when monitoring performance
        """
        if self.training_configuration.training_dictionary['loss'] == 'MSE':
            loss_type = tf.keras.losses.MeanSquaredError()
            metric = ['mae']
        elif self.training_configuration.training_dictionary['loss'] == 'MAE':
            loss_type = tf.keras.losses.MeanAbsoluteError()
            metric = ['mae']
        elif self.training_configuration.training_dictionary['loss'] == 'ENTROPY':
            loss_type = tf.keras.losses.CategoricalCrossentropy()
            metric = 'accuracy'
        else:
            print('Unrecognized loss function: please select from MSE, MAE, or ENTROPY.')
            loss_type = {}
            metric = ''
            exit(-1)
        return loss_type, metric

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
            model = []
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

        # Provide a model summary
        keras_model.summary()

        # Plot the graph
        if self.training_configuration.training_dictionary['plot_network']:
            keras.utils.plot_model(keras_model, "model_to_train.png", show_shapes=True)

        # Create a dataset
        training_dataset = self.create_dataset(self.training_data[0][0], self.training_data[0][1])
        validation_dataset = self.create_dataset(self.training_data[1][0], self.training_data[1][1])

        # Create an iterator
        # training_iter = training_dataset.__iter__()
        # validation_iter = validation_dataset.__iter__()
        # x_train, y_train = training_iter.get_next()
        # x_valid, y_valid = validation_iter.get_next()

        # Define a loss and metric
        loss, metric = self.define_loss_and_metric()

        # Compile the model
        keras_model.compile(
            optimizer=self.training_configuration.training_dictionary['optimizer'],
            loss=loss,
            metrics=metric
        )

        history = keras_model.fit(
            training_dataset,
            steps_per_epoch=self.n_training_samples
            // self.training_configuration.training_dictionary['batch_size'],
            epochs=self.training_configuration.training_dictionary['epochs'],
            validation_data=validation_dataset,
            validation_steps=self.n_validation_samples
            // self.training_configuration.training_dictionary['batch_size'] + 1,
        )

        # Plot the results
        history_keys = list(history.history.keys())
        if self.training_configuration.training_dictionary['plot_curve']:
            plot_results(history, history_keys, self.training_configuration.training_dictionary)

        # Save model
        if self.training_configuration.training_dictionary['save_model']:
            keras.models.save_model(keras_model,
                                    self.training_configuration.network_dictionary['model_name']+'.h5')


if __name__ == '__main__':
    train_ai = TrainAI(sys.argv[1])

    train_ai.train_model()
