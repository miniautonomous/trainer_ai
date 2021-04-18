"""
  Decription:
    Script to run the a trained Keras.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from h5IO import h5Files
from tsDSP import tsDSP

# Set the correct GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# DNN file
dnnPath = 'C:\\Users\\fcharett\\docsWork\\projects\\prjFollowTheLeader\\devInfer\\fordIPU\\'
# dnnFile = '200428.184211_4C_1R_regression.h5'
dnnFile = 'FordRegression_Keras_IPU_Single_Day_47_84.h5'
# dnnFile = 'FordRegression_Keras_IPU_All_Days_47_84.h5'

nnModel = tf.keras.models.load_model(dnnPath+dnnFile, custom_objects={"tf":tf})
nnModel.summary()
# First retrieve the model input sizes
tmpCfg = nnModel.get_config()
rnnSeqLength = tmpCfg['layers'][0]['config']['batch_input_shape'][1]
imgHeight = tmpCfg['layers'][0]['config']['batch_input_shape'][2]
imgWidth = tmpCfg['layers'][0]['config']['batch_input_shape'][3]
inNumChannels = tmpCfg['layers'][0]['config']['batch_input_shape'][4]


# Create an io object to process the data file
io = h5Files() # Create a "h5IO" object
# "test" file
testPath = 'C:\\Users\\fcharett\\docsWork\\projects\\prjFollowTheLeader\\devInfer\\'
testFile = '200423_165823_xCar_numImg02_CW.hdf5'
print(f'Processing file => {testPath+testFile}')
# Get current the file info
io.fileInfo(testPath+testFile)
# Find the number of frames/images in the current file
nEntries = len(io.groupList)
# Initialize the image buffer for the current file
imageCatalog = np.zeros((nEntries, int(io.fileAtt['imgHeight']), int(io.fileAtt['imgWidth']), 6),
                        np.uint8)

# Read ALL the images in the file and put them in the image buffer
for j in range(len(io.groupList)):
  imgTmp = io.h5DataSetImage(io.groupList[j], 'imgLeft')
  for k in range(1, 2):
      # If there are multiple images in each frame or group, concatenate them
    imgTmp = np.concatenate((imgTmp, io.h5DataSetImage(io.groupList[j], 'imgRight')), axis=2)
  imageCatalog[j] = imgTmp
#=================================== Initialize the ts lists ======================================#
spRef = tsDSP(None)
spY = tsDSP(None)
# Build the new timeseries names AND Get the current file targets data, i.e. ground
# truth for regression
predictNames = ['Predict - steering']
spRef.appendTs(io.tsFromGroupScalar('steering', testPath+testFile))
# Create the new prediction timeseries
spY.new(predictNames)
#==================================== Perform ALL the inferences ==================================#
# Need to create a buffer for the sequence
imgIn = np.zeros((1, rnnSeqLength, int(imgHeight), int(imgWidth), int(inNumChannels)), np.uint8)
# Loop around ALL the images in the file building sequence at each images
for iImg in range(0, len(imageCatalog)-rnnSeqLength):
  if iImg % 50 == 0:
    print(f'processing frame => {iImg}')
  imgIn[0, :, :, :, :] = imageCatalog[iImg:iImg+rnnSeqLength, :, :, :]
  # Save to the ts
  spY.addNewDataPoint(nnModel.predict(imgIn)[0][0], 'Predict - steering')
#============================ Compute AND save the results/plots to file ==========================#
groundTruth = spRef.getY('steering')[rnnSeqLength:]
resDiff = groundTruth-spY.getY(f'Predict - steering')
mse = np.sqrt(np.mean(resDiff**2))
ampRange = spRef.ts[0].prop["max"]-spRef.ts[0].prop["min"]
# Build the plot title
textTitle = f'Test File Name => {testFile}\nMSE (%) => {100*mse/ampRange:2.1f}'
#---------------------------Create the associated plot file. i.e. png------------------------------#
plt.figure(figsize=(12, 8))
plt.plot(groundTruth, 'g', linewidth=4.0)
plt.plot(spY.getY('Predict - steering'), 'r-')
plt.ylabel('Normalized Steering Angle')
plt.xlabel('Frame Index')
plt.grid(True)
plt.legend([f'Ground Truth - {spRef.ts[0].prop["name"]}', spY.ts[0].prop['name']], loc='best')
plt.ylim(-100, 100)
plt.title(textTitle)
plt.savefig(f'{dnnPath+time.strftime("%y%m%d" + "." + "%H%M%S")+testFile[:-5]}_steering.png')