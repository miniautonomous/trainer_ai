import glob
from utils.data_loader import BatchLoader
import cv2


"""
    File: movie_maker.py

    Purpose: 
        Creates a movie file from an hdf5 file recorded from the vehicle's
        perspective as it is driving itself in autonomous mode.
"""

# Specs of your video
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 30
image_width = 120
image_height = 90

# HDF5 file that has the recording
raw_movie_directory = '../data/'
raw_movie_files = glob.glob(raw_movie_directory+'*.hdf5')
print(f'Number of recorded hdf5 files: '+str(len(raw_movie_files)))

# Create dummy entries for the two dictionaries that are required for the batch loader
network_dictionary = {'image_width': 120,
                      'image_height': 90,
                      'throttle': True,
                      'sequence': True,
                      'sequence_length': 5,
                      'sequence_overlap': 2}
data_dictionary = {'data_directory': 'test_path',
                   'shuffle': False,
                   'train_to_valid': 0.85,
                   'normalize': True}

# Create the data loader
data_loader = BatchLoader(data_dictionary,
                          network_dictionary,
                          'regression')

for raw_movie_file in raw_movie_files:
    video_file = cv2.VideoWriter(raw_movie_file+'.mp4', codec, float(fps), (image_width, image_height))
    # Use the batch loader
    image_data, reference_labels = data_loader.read_data_file(raw_movie_file)

    # Number of images and labels
    number_entries = len(image_data)

    # Go through the frames of the HDF5 file and create a video
    for i in range(number_entries):
        video_file.write(image_data[i])
    video_file.release()
