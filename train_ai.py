import numpy as np
import string
import sys
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# GPU identifier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# If multiple CUDA compatible devices are available,
# you can select an index other than 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainAI(object):
    def __init__(self, input_confg: string = None):
        """
        Trains the models to be deployed to MiniAutonomous!

        Parameters
        ----------
        input_confg: (string) configuration file that defines the training process
        """

    def train_model(self):
        print("Need to be code this.")


if __name__ == '__main__':
    train_ai = TrainAI(sys.argv[1])

    # Train the model
    train_ai.train_model()