import time
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils.data_loader import BatchLoader
import glob

USE_TRT = True
SAVE_FIG = True

"""
    File: dnn_inference.py
    
    Purpose: 
        Run inference on a test data set to quantify performance.
"""

# GPU identifier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print(device_lib.list_local_devices())
# If multiple CUDA compatible devices are available,
# you can select an index other than 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DNN File
dnn_path = './model_files/'
if USE_TRT:
    dnn_file = 'garage_stateless_bs16_epochs_100/'

    # Load the model
    nn_model = tf.saved_model.load(dnn_path + dnn_file)
    prediction = nn_model.signatures['serving_default']
    if len(prediction.inputs[0].shape) == 5:
        has_sequence = True
        sequence_length = prediction.inputs[0].shape[1]
        image_height = prediction.inputs[0].shape[2]
        image_width = prediction.inputs[0].shape[3]
        channel_depth = prediction.inputs[0].shape[4]
    else:
        has_sequence = False
        sequence_length = 1
        image_height = prediction.inputs[0].shape[1]
        image_width = prediction.inputs[0].shape[2]
        channel_depth = prediction.inputs[0].shape[3]
else:
    dnn_file = 'GarageLoopModel_bs16.h5'

    # Load the model
    nn_model = tf.keras.models.load_model(dnn_path + dnn_file,
                                          custom_objects={"tf": tf})
    nn_model.summary()
    # First retrieve the model input sizes
    model_config = nn_model.get_config()
    # We have a model with state memory (i.e. contains an LSTM, GRU, etc.)
    if len(model_config['layers'][0]['config']['batch_input_shape']) == 5:
        has_sequence = True
        sequence_length = model_config['layers'][0]['config']['batch_input_shape'][1]
        image_height = model_config['layers'][0]['config']['batch_input_shape'][2]
        image_width = model_config['layers'][0]['config']['batch_input_shape'][3]
        channel_depth = model_config['layers'][0]['config']['batch_input_shape'][4]
    else:
        has_sequence = False
        sequence_length = 1
        image_height = model_config['layers'][0]['config']['batch_input_shape'][1]
        image_width = model_config['layers'][0]['config']['batch_input_shape'][2]
        channel_depth = model_config['layers'][0]['config']['batch_input_shape'][3]

# Test all files in a directory
test_path = './test_files/'
test_files = glob.glob(test_path+'*.hdf5')
print(f'Number of test files: '+str(len(test_files)))

# Create config to load data
network_dictionary = {'image_width': image_width,
                      'image_height': image_height,
                      'throttle': True,
                      'sequence': has_sequence,
                      'sequence_length': sequence_length,
                      'sequence_overlap': 2}
data_dictionary = {'data_directory': test_path,
                   'shuffle': False,
                   'train_to_valid': 0.85,
                   'normalize': True}

# Create the data loader
data_loader = BatchLoader(data_dictionary,
                          network_dictionary,
                          'regression')
for test_file in test_files:
    # Read image and label data from the test file
    image_data, reference_labels = data_loader.read_data_file(test_file)

    # Number of images and labels
    number_entries = len(image_data)

    # Predicted label array
    predicted_steering = np.empty([0])
    predicted_throttle = np.empty([0])

    # Need to create a buffer for the sequence
    if has_sequence:
        image_in = np.zeros((1,
                             sequence_length,
                             int(image_height),
                             int(image_width),
                             int(channel_depth)),
                            np.uint8)
    else:
        image_in = np.zeros((1,
                             int(image_height),
                             int(image_width),
                             int(channel_depth)),
                            np.uint8)
    # Loop around ALL the images in the file building sequence at each images
    for image_index in range(0, len(image_data) - sequence_length):
        if image_index % 50 == 0:
            print(f'processing frame => {image_index}')
        if has_sequence:
            image_in[0, :, :, :, :] = image_data[image_index:image_index + sequence_length, :, :, :]
        else:
            image_in[0, :, :, :] = image_data[image_index, :, :, :]

        # Perform inference
        if USE_TRT:
            inference = prediction(tf.convert_to_tensor(image_in, dtype=tf.float32))
            inference = inference['dense'][0].numpy()
        else:
            inference = nn_model.predict(image_in)[0]

        if has_sequence:
            predicted_steering = np.append(predicted_steering, inference[-1][0])
            predicted_throttle = np.append(predicted_throttle, inference[-1][1])
        else:
            predicted_steering = np.append(predicted_steering, inference[0])
            predicted_throttle = np.append(predicted_throttle, inference[1])

    # Compute RMSE and save the results
    # Steering
    ground_truth_steering = reference_labels[sequence_length:, 0]
    residual_difference_steering = ground_truth_steering - predicted_steering[0]
    rmse_steering = np.sqrt(np.mean(residual_difference_steering ** 2))
    amplitude_range_steering = 200

    # Throttle
    ground_truth_throttle = reference_labels[sequence_length:, 1]
    residual_difference_throttle = ground_truth_throttle - predicted_throttle[1]
    rmse_throttle = np.sqrt(np.mean(residual_difference_throttle ** 2))
    amplitude_range_throttle = 100

    # Build the plot title
    text_title = f'Test File Name => {test_file}\n RMSE Steering (%) => ' \
                 f'{100 * rmse_steering / amplitude_range_steering:2.1f}  ' \
                 'RMSE Throttle (%) =>' \
                 f'{100 * rmse_throttle / amplitude_range_throttle:2.1f}'

    # Create a plot
    plt.figure(figsize=(12, 8))
    plt.plot(ground_truth_steering, 'g', linewidth=4.0)
    plt.plot(predicted_steering, 'r-')
    plt.plot(ground_truth_throttle, 'k', linewidth=4.0)
    plt.plot(predicted_throttle, 'b--')
    plt.ylabel('Normalized Steering Angle and Throttle', fontsize=16)
    plt.xlabel('Frame Index', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Ground Truth - Steering',
                'Inference - Steering',
                'Ground Truth - Throttle',
                'Inference - Throttle'], loc='best', fontsize=14)
    plt.ylim(-100, 100)
    plt.title(text_title, fontsize=18)
    if SAVE_FIG:
        if USE_TRT:
            plt.savefig(dnn_path+dnn_file+f'{test_file[13:]}_simulation.png')
        else:
            plt.savefig(dnn_path+f'{test_file[13:]}_simulation.png')
    plt.show()
