"""
  Decription:
    Script to run the a trained Keras.
"""
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils.data_loader import BatchLoader
# from tsDSP import tsDSP

# GPU identifier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print(device_lib.list_local_devices())
# If multiple CUDA compatible devices are available,
# you can select an index other than 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DNN File
dnn_path = './model_files/'
dnnFile = 'TestModel.h5'

# Test File
test_path = './data_files/'
test_file = 'TestData.hdf5'

# Load the model
nn_model = tf.keras.models.load_model(dnn_path + dnnFile,
                                      custom_objects={"tf": tf})
nn_model.summary()
# First retrieve the model input sizes
model_config = nn_model.get_config()
sequence_length = model_config['layers'][0]['config']['batch_input_shape'][1]
image_height = model_config['layers'][0]['config']['batch_input_shape'][2]
image_width = model_config['layers'][0]['config']['batch_input_shape'][3]
channel_depth = model_config['layers'][0]['config']['batch_input_shape'][4]

# Create a data dictionary to read the file
data_dictionary = {'image_width': image_width,
                   'image_height': image_height,
                   'data_directory': test_path,
                   'shuffle': False,
                   'throttle': True,
                   'train_to_valid': 0.85,
                   'sequence': True,
                   'sequence_length': 5,
                   'sequence_overlap': 2,
                   'normalize': False}

# Create the data loader
data_loader = BatchLoader(data_dictionary,
                          'regression')

# Read image and label data from the test file
image_data, reference_label = data_loader.read_data_file(test_path + test_file)

# Number of images and labels
number_entries = len(image_data)
#=================================== Initialize the ts lists ======================================#
# reference_label = tsDSP(None)
spY = tsDSP(None)
# Build the new time series names AND Get the current file targets data, i.e. ground
# truth for regression
prediction_name = ['Predict - steering']
# Create the new prediction time series
spY.new(prediction_name)
#==================================== Perform ALL the inferences ==================================#
# Need to create a buffer for the sequence
image_in = np.zeros((1,
                     sequence_length,
                     int(image_height),
                     int(image_width),
                     int(channel_depth)),
                    np.uint8)
# Loop around ALL the images in the file building sequence at each images
for image_index in range(0, len(image_data) - sequence_length):
    if image_index % 50 == 0:
        print(f'processing frame => {image_index}')
    image_in[0, :, :, :, :] = image_data[image_index:image_index + sequence_length, :, :, :]
    # Save to the ts
    spY.addNewDataPoint(nn_model.predict(image_in)[0][0], 'Predict - steering')
#============================ Compute AND save the results/plots to file ==========================#
groundTruth = reference_label.getY('steering')[sequence_length:]
resDiff = groundTruth-spY.getY(f'Predict - steering')
mse = np.sqrt(np.mean(resDiff**2))
ampRange = reference_label.ts[0].prop["max"] - reference_label.ts[0].prop["min"]
# Build the plot title
textTitle = f'Test File Name => {test_file}\nMSE (%) => {100 * mse / ampRange:2.1f}'
#---------------------------Create the associated plot file. i.e. png------------------------------#
plt.figure(figsize=(12, 8))
plt.plot(groundTruth, 'g', linewidth=4.0)
plt.plot(spY.getY('Predict - steering'), 'r-')
plt.ylabel('Normalized Steering Angle')
plt.xlabel('Frame Index')
plt.grid(True)
plt.legend([f'Ground Truth - {reference_label.ts[0].prop["name"]}', spY.ts[0].prop['name']], loc='best')
plt.ylim(-100, 100)
plt.title(textTitle)
plt.savefig(f'{dnn_path + time.strftime("%y%m%d" + "." + "%H%M%S") + test_file[:-5]}_steering.png')