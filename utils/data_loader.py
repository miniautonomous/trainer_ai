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

        # Set data type for labels
        if mode == 'regression':
            self._d_type = np.float32
        else:
            self._d_type = np.int32

        # Initiate numpy array for training images and training labels based on desired config
        self.training_images = np.zeros((0, self._data_config['sequence_length'],
                                         self._data_config['image_height'],
                                         self._data_config['image_width'],
                                         3), dtype=np.float32)
        self.validation_images = np.zeros((0, self._data_config['sequence_length'],
                                           self._data_config['image_height'],
                                           self._data_config['image_width'],
                                           3), dtype=np.float32)

        if self._data_config['sequence'] and self._data_config['throttle']:
            self.training_labels = np.zeros((0, self._data_config['sequence_length'], 2),
                                            self._d_type)
            self.validation_labels = np.zeros((0, self._data_config['sequence_length'], 2),
                                              self._d_type)
        elif self._data_config['sequence'] and not self._data_config['throttle']:
            self.training_labels = np.zeros((0, self._data_config['sequence_length']),
                                            self._d_type)
            self.validation_labels = np.zeros((0, self._data_config['sequence_length']),
                                              self._d_type)
        else:
            self.training_labels = np.zeros((0,), self._d_type)
            self.validation_labels = np.zeros((0,), self._d_type)

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
            temp_image = np.zeros((len(group_list), int(file_attributes['imgHeight']),
                                   int(file_attributes['imgWidth']), 3), np.uint8)

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
                temp_image[index, :, :, 0:3] = np.array(hf[item]['image'])

                # Read label
                if self._mode == 'regression' and self._data_config['throttle']:
                    temp_label[index, 0] = np.array(hf[item]['steering'])
                    temp_label[index, 1] = np.array(hf[item]['throttle'])
                elif self._mode == 'regression' and not self._data_config['throttle']:
                    temp_label[index] = np.array(hf[item]['steering'])
                else:
                    temp_label[index] = np.argmax(hf[item]['classIndex'])

        return temp_image, temp_label

    def load_from_hdf5(self) -> [np.ndarray, np.ndarray]:
        """
            Read individual data frames from all the HDF5 files
            found in a directory.

        Returns
        -------
        training dataset: ([np.ndarray, np.ndarray]) training and validation data
        """
        for count, file in enumerate(self._files):
            print('Reading file {}: {}/{}'.format(file, count+1, len(self._files)))

            # Read the data from the file
            temp_image, temp_label = self.read_data_file(file)

            # Finished current file so collate with general data
            self.training_images = np.concatenate((self.training_images, temp_image), axis=0)
            self.training_labels = np.concatenate((self.training_labels, temp_label), axis=0)

        # Done reading files; shuffle if required
        """
            Here each data frame is an independent entity, so treat it as a jumbled data blob and 
            shuffle individual frames indiscriminately across files
        """
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

    def load_sequence_from_hdf5(self) -> [np.ndarray, np.ndarray]:
        """
            Load a sequence of frames from HDF5.

        Returns
        -------
        training dataset: ([np.ndarray, np.ndarray]) training and validation data
        """
        for count, file in enumerate(self._files):
            print('Reading file {}: {}/{}'.format(file, count + 1, len(self._files)))

            # Read the data from the file
            temp_image, temp_label = self.read_data_file(file)

            # Determine the number of frames in the given file
            number_of_frames = len(temp_image)

            # Create a valid index list for sequences
            valid_starting_indices = list(range(number_of_frames -
                                                self._data_config['sequence_length']))

            """
                Since we are allowing sequences to have overlaps, we need to make an adjustment
                to how the file data frames are indexed. Let's say we are considering the following
                two sequences of five data frames, where the number are the data frame index:
                
                Sequence A: [30 31 32 33 34]
                Sequence B: [35 36 37 38 39]
                
                In the above, we are not allowing for an overlap, which would not only waste valid
                sequences but also produce a training set correlated to a single timeline across the 
                data file. It would be better to allow for individual frame reuse, as in:
                
                Sequence A: [30 31 32 33 34]
                Sequence B: [32 33 34 35 36]
                
                Now we have reused three frames, (32, 33 and 34). We add the parameter, 'sequence_overlap',
                to define how many frames of overlap are permitted between distinct sequences.
            """
            sequence_offset = self._data_config['sequence_length'] - self._data_config['sequence_overlap']
            valid_starting_indices = valid_starting_indices[::sequence_offset]
            """
                The above does this:
                list_example = [1, 2, 3, 4, 5, 6]
                list_example = [::2]
                list_example = [1, 3, 5]
            """

            # Now we we can proceed to extract sequences, but shuffle if necessary
            if self._data_config['shuffle']:
                valid_starting_indices = shuffle(valid_starting_indices)

            # Determine the training and validation split
            total_number_sequences = len(valid_starting_indices)
            samples_for_training = int(self._data_config['train_to_valid'] * total_number_sequences)
            training_indices = valid_starting_indices[0:samples_for_training]
            validation_indices = valid_starting_indices[samples_for_training:]

            # Do actual data set creation
            training_frames = []
            training_labels = []
            validation_frames = []
            validation_labels = []
            # Training data
            for starting_index in training_indices:
                training_frames.append(temp_image[starting_index:
                                                  starting_index+self._data_config['sequence_length']])
                training_labels.append(temp_label[starting_index:
                                                  starting_index+self._data_config['sequence_length']])

            # Validation data
            for starting_index in validation_indices:
                validation_frames.append(temp_image[starting_index:
                                                    starting_index+self._data_config['sequence_length']])
                validation_labels.append(temp_label[starting_index:
                                                    starting_index+self._data_config['sequence_length']])

            # Finished processing the given file, so concatenate with overall data sets
            self.training_images = np.concatenate((self.training_images, np.asarray(training_frames)), axis=0)
            self.training_labels = np.concatenate((self.training_labels, np.asarray(training_labels)), axis=0)
            self.validation_images = np.concatenate((self.validation_images, np.asarray(validation_frames)), axis=0)
            self.validation_labels = np.concatenate((self.validation_labels, np.asarray(validation_labels)), axis=0)

        # Package the data
        training_data = [self.training_images, self.training_labels]
        validation_data = [self.validation_images, self.validation_labels]

        return [training_data, validation_data]
