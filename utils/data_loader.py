import glob
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
