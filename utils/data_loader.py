import numpy as np
import glob
import h5py
from sklearn.utils import shuffle
import string


class BatchLoader:
    def __init__(self, data_dictionary: dict, mode: string):
        """
            Batch processing utility that reads HDF5 data files and creates
            training and validation data sets that can be accessed at
            specified batch sizes.

        Parameters
        ----------
        data_dictionary: (dict) configuration of the data
        mode: (string) regression or classification data
        """
        # Read the data configuration
        self._data_config = data_dictionary

        # Classifier or regression?
        self._mode = mode

        # Set data type
        if mode == 'regression':
            self.d_type = np.float32
        else:
            self.d_type = np.int32

        # Initiate numpy array for training images and training labels based on desired config
        self.training_images = np.zeros((0, self._data_config['sequence_length'],
                                         self._data_config['height'],
                                         self._data_config['width'],
                                         3), dtype=np.float32)

        if self._data_config['sequence'] and self._data_config['throttle']:
            self.training_labels = np.zeros((0, self._data_config['sequence_length'], 2),
                                            self.d_type)
        elif self._data_config['sequence'] and not self._data_config['throttle']:
            self.training_labels = np.zeros((0, self._data_config['sequence_length']),
                                            self.d_type)
        else:
            self.training_labels = np.zeros((0,), self.d_type)

        # Create a list of all the HDF5 in the data directory
        self._files = glob.glob(self._data_config['data_directory'])

        # Zero out important indices
        self.samples_for_training = 0
        self.samples_for_validation = 0
        self.train_index = 0
        self.val_index = 0

    def read_data_file(self, file: string) -> [np.ndarray, np.ndarray]:
        """
            Read data from a specified HDF5 file

        Parameters
        ----------
        file: (string) HDF5 input data file

        Returns
        -------
        temp_image, temp_label: ([np.ndarray, np.ndarray]) image and corresponding label for
                                training

        """
        # File specific parameters
        file_attributes = {}
        group_list = []
        group_dataset = []

        # Process current file
        with h5py.File(file, 'r') as hf:
            # 1. Screen the file attributes
            for item in hf.attrs.keys():
                if str(type(hf.attrs[item])) == "<class 'str'>" or \
                        str(type(hf.attrs[item])) == "<class 'numpy.ndarray'>" or \
                        str(type(hf.attrs[item])) == "class 'numpy.float64'>":
                    file_attributes[item] = hf.attrs[item]
                else:
                    file_attributes[item] = hf.attrs[item].decode()

            # 2. Screen through the groups of the file
            for item in hf:
                if isinstance(hf[item], h5py.Group):
                    group_list.append(item)

            # 3. Specify what data exists within a group
            for item in hf[group_list[0]].items():
                group_dataset.append(item[0])

            # Create an image array
            # TODO: Verify the data type!
            temp_image = np.zeros((len(group_list), int(file_attributes['imageHeight']),
                                   int(file_attributes['imageWidth']), 3), np.uint8)

            # Define what type of label is appropriate
            if self._mode == 'regression' and self._data_config['throttle']:
                temp_label = np.zeros((len(group_list), 2), np.float32)
            elif self._mode == 'regression' and not self._data_config['throttle']:
                temp_label = np.zeros((len(group_list),), np.float32)
            else:
                temp_label = np.zeros((len(group_list),), np.uint8)

            # Go through the group list and do the actual reading
            for index, item in enumerate(group_list):
                # Read image
                # TODO: discuss this with Francois!
                temp_image[index, :, :, 0:3] = np.array(hf[item]['imgLeft'])

                # Read label
                if self._mode == 'regression' and self._data_config['throttle']:
                    temp_label[index, 0] = np.array(hf[item]['steering'])
                    temp_label[index, 1] = np.array(hf[item]['throttle'])
                elif self._mode == 'regression' and not self._data_config['throttle']:
                    temp_label[index, 0] = np.array(hf[item]['steering'])
                else:
                    temp_label[index, 0] = np.argmax(hf[item]['classIndex'])

        return temp_image, temp_label

    def load_from_hdf5(self) -> [np.ndarray, np.ndarray]:
        """
            Read individual data frames from all the HDF5 files
            found in a directory.

        Returns
        -------
        training_dataset: ([np.ndarray, np.ndarray]) training and validation data
        """
        for count, file in enumerate(self._files):
            print('Reading file {}: {}/{}'.format(file, count+1, len(self._files)))

            # Read the data from the file
            temp_image, temp_label = self.read_data_file(file)

            # Finished current file so collate with general data
            self.training_images = np.concatenate((self.training_images, temp_image), axis=0)
            self.training_labels = np.concatenate((self.training_labels, temp_label), axis=0)

        # Done reading files; shuffle if required
        if self._data_config['shuffle']:
            self.training_images, self.training_labels = shuffle(self.training_images,
                                                                 self.training_labels)

        # Do the training and validation split
        total_number_frames = len(self.training_labels)
        samples_for_training = int(self._data_config['train_to_valid']*total_number_frames)
        training_data = (self.training_images[0:samples_for_training],
                         self.training_labels[0:samples_for_training])
        validation_data = (self.training_images[samples_for_training:],
                           self.training_labels[samples_for_training:])
        return [training_data, validation_data]
