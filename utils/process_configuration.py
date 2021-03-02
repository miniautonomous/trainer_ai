import configparser
import string
import sys


class ConfigurationProcessor:
    """
        Process the training configuration for the training session.

    Input configuration files have three sections:
        1) Training: specifications of the training session (e.g. epochs, val/train ratio, etc.)
        2) Network: model definition and associated parameters
        3) Data: specifications and characteristics involving the training data

    """

    def __init__(self, input_file: string = None):
        """
        Parameters
        ----------
        input_file:  (string) configuration input file

        """
        # Dictionaries defined in the input file
        self.training_dictionary = {}
        self.network_dictionary = {}
        self.data_dictionary = {}

        # Parser object
        self.parser = configparser.ConfigParser()

        # Process the file
        if input_file is None:
            sys.exit('Please provide an input configuration file.')
        else:
            self.parser.read(input_file)

        # Process the desired configuration
        self._process_configuration()

    def _process_configuration(self):
        for section in self.parser.sections():

            # Training section
            if section == 'Training':
                integer_list = ['decay_steps', 'epochs', 'batch_size']
                for option in self.parser.options(section):
                    if option in integer_list:
                        self.training_dictionary[option] = \
                            self.parser.getint(section, option)
                    elif option == 'starting_learning_rate':
                        self.training_dictionary[option] = \
                            self.parser.getfloat(section, option)
                    else:
                        self.network_dictionary[option] = \
                            self.parser.get(section, option)

            # Network section
            if section == 'Network':
                for option in self.parser.options(section):
                    self.network_dictionary[option] = \
                        self.parser.get(section, option)

            # Data section
            if section == 'Data':
                integer_list = ['image_height', 'image_width',
                                'sequence_length', 'sequence_overlap']
                boolean_list = ['shuffle', 'throttle', 'sequence']
                for option in self.parser.options(section):
                    if option in integer_list:
                        self.data_dictionary[option] = \
                            self.parser.getint(section, option)
                    elif option in boolean_list:
                        self.data_dictionary[option] = \
                            self.parser.getboolean(section, option)
                    elif option == 'train_to_valid':
                        self.data_dictionary[option] = \
                            self.parser.getfloat(section, option)
                    else:
                        self.data_dictionary[option] = \
                            self.parser.get(section, option)