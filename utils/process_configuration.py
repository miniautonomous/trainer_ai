import configparser
import string
import sys


class ConfigurationProcessor:
    """
        Process the training configuration for the training session.

    Input configuration files have three sections:
        1) Training: specifications of the training session (e.g. epochs, val/train ratio, etc.)
        2) Network: Model definition and associated parameters
        3) Data: specifications and characteristics involving the training data
    """

    def __init__(self, input_file: string = None):
        """
        input_file: (string) configuration input file that controls the training session

        Parameters
        ----------
        input_file:  (string) configuration input file
        """
        # Dictionaries defined in the input file
        self.trainin_dictionary = {}
        self.network_dictionary = {}
        self.data_dictionary = {}

        # Parser object
        self.parser = configparser.ConfigParser()

        # Process the file
        print('pause')
