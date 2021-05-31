# trainer_ai

Welcome to the code base for *trainer_ai*!

This is the software stack is training counterpart of *engine_ai*, (https://github.com/miniautonomous/engine_ai), and 
allows you to train network models for inference for the **MiniAutonomous** vehicle. In this README, we are going to 
review the basic installation process, the overall code structure, and then review a few key facets to consider when 
training network models for end-to-end driving applications. 

Please be aware that additional supporting content is available in our main portal page:

https://miniautonomous.github.io/portal/

This is where you will find more resources regarding *engine_ai*, (the on-vehicle stack), vehicle assembly videos, and
a variety of other information that might be helpful on you journey. Enjoy!

# Table of Contents

1. [Installation](#installation)

2. [Code Structure](#code-structure)

3. [Usage](#usage)

4. [Simulation is the Key to Success](#simulation-is-the-key-to-success)

5. [Training End-to-End Networks](#training-end-to-end-networks)

    i. [Speed is Also Key](#speed-is-also-key)

    ii. []

# Installation

As opposed to *engine_ai*, where one is constrained by certain limitations of the Jetson Nano platform, getting 
*trainer_ai* up and running should be a straightforward process. We have a provided a *requirements.txt* file, which
once Cuda and CuDNN are installed should be able to be installed by

```angular2html
    pip3 install -r requirements.txt
```

As of the writing of this README, we are using CUDA Version 10.2.89 and CuDNN 7. 

The key element to focus on is that the Jetson Nano is flashed with *JetPack 4.5*, which allows you to install 
*Tensorflow* 2.3.1. You may notice that our *requirements.txt* file has *Tensorflow* 2.2.0.  Whereas you are generally 
constrained what *Tensorflow* you can install on the Nano, we have anecdotally found that using a *Tensorflow* 2.1+ is
generally compatible with our the training stack.

# Code Structure

The focal script is *trainer_ai.py*, which will allow you to load from your vehicle, define a network topology for the 
end-to-end autonomous application, save the resulting model to either a **Keras** model file or a **TensorRT** parsed 
model directory, and finally run inference on test files to simulate how well the network will do on the chosen task. 
All these topics are review in the following sections.

Here is a brief summary of the primary files and directories' content to orient you as you study the code.

```angular2html
    trainer_ai.py: primary script that loads data from distinct files and creates *Tensorflow* a **dataset**, creates
                   a training configuration in terms of defining a loss and metric for accuracy, compiles a model and 
                   trains it using the **Keras** **fit** method.
    >[custom_layers]: directory to define custom layer definitions for use in novel network models
    >[models]: directory that contains network model definitions
    >[sample_input_scripts]: directory that contains various examples of input scripts to pass to trainer_ai.py to 
                             define a training run
    >[simulation]: directory that contains the file dnn_inference.py, that allows the user to run an inference test on 
                   test data.
    >[utils]: directory that contains various utilities that help load data, process input scripts and plot training and
              simulation results.
```

# Usage

Great, so how do we use it? Right, to kick things off you can use any of the input scripts present in 
**sample_input_scripts** as a template and once redefined as needed, type the following

```angular2html
    python3 trainer_ai.py input_script.txt
```

The key here is the content of **input_script.txt**. Two examples are given in the 'sample_input_script' directory: one 
that can be used to train a network with state memory, (i.e. contains a GRU, LSTM, etc.), and one that can train a 
stateless model, (i.e. image in, inference out). They are both similar in nature and have the same key template 
consisting of three distinct sections, each with their own distinct options. Below is the input template:

```angular2html
    >[Training]:
        > optimizer: (str) name of desired optimizer -- use any made available by Tensorflow/keras), but for standard 
                     use cases 'rmsprop' does well
        > loss: (str) desired loss function -- options are 'MSE', 'MAE', and 'ENTROPY'
        > epochs: (int) number of epochs to train for
        > batch_size: (int) size of mini-batch for each gradient evaluation
        > plot_network: (bool) plot a graph depiction of the network being trained?
        > plot_curve: (bool) plot resulting loss/accuracy curve after training
        > save_curve: (bool) save the plot?
        > save_model: (bool) do you wish to save the resulting model? (default save is to Keras HDF5 file)
        > decay_type: PLACE HOLDER --  We have not had to alter standard decay elements for training, but we have left
                                       this parameter here in case future models require these specs
        > starting_learning_rate: PLACE HOLDER
        > decay_steps: PLACE HOLDER

    >[Network]:
        > model_name: (str) Name as it model name appears in 'Model' directory. (File and class name should match.)
        > precision: (str) Numeric precision of master weights, options are 'half' or 'single'
        > save_to_trt: (bool) parse the resulting model via 'TensorRT'
        > image_width: (int) width of the input tensor into the network
        > image_height: (int) height of the input tensor into the network
        > sequence: (bool) does the network being trained require a sequence of images, (i.e. it has a GRU, LSTM, etc, 
                    needs a sequence of images to train)
        > sequence_length: (int) number of frames that need to be in a sequence to train
        > sequence_overlap: (int) number of frames that can be shared across distinct sequences, (e.g. if you are using
                            frames [1, 2, 3, 4] and you have an overlap of 2, then you can also uses frames [3, 4, 5, 6]
                            for training)
        > throttle: (bool) are we training a network that will have an output for throttle?

    >[Data]
        > data_directory: (str) directory where the data that was uploaded from the car and will be used for training is
                          stored
        > shuffle: (bool) do you wish to shuffle the data?
        > large_data: (bool) is the data too large, (beyond 2 GBs), to create a single 'Tensorflow' dataset. If so, 
                      attempt to append distinct datasets into a concatenation of datasets. (See lines 77 to 85 of 
                      trainer_ai.py.) 
        > train_to_valid: (float) ratio of training-to-validation data to use while training
        > normalize: (bool) Do you wish to normalize the data from -100 to 100 for steering and 0 to 100 for throttle or
                     use raw PWM values. (The latter, although possible, is not advisable.)
```

If you wish to make changes to the input template, review *utils/process_configuration.py* to see how input scripts are
parsed for a training session.

# Simulation is the Key to Success

It would be a very painful iteration process to train a model and then deploy it back to the vehicle without any sense 
of how the model should theoretically do in practice. Having a poor validation curve gives you some sense, but for 
various applications, it is much clearer to have a series of test data files that are extracted from the training data 
which are used to run as input for pure inference quality tests. 

This functionality is provided in 'simulation/dnn_inference.py'. Please review the script now. It is quite 
straightforward. Once you specify an input model, either saved as **Keras** model file or a *TensorRT* model directory, 
you then specify the location of the test files, (line 71), and then you can run an inference test on the data to see 
how close your model is to the manual input of the driver. An example result for steering is given below. (The throttle 
plot has been commented out, but uncomment lines 162-163 and you will see the throttle plot.)

<p align="center">
<img src=./img/simulation_example.png width="75%"><p></p>
<p align="center"> Figure 1: Example of a simulation plot</p>

Based on our experience, normally steering RMSE's on the order of 20% or less will prove somewhat reliable, while under 
10-12% will be very accurate. We scale the RMSE by the total normalized scale of steering available. Throttle is more 
difficult to pinpoint and is much more correlated to the autonomous objective in question.

# Training End-to-End Networks

There are a number of elements to consider when training networks for various end-to-end applications, and hopefully 
*trainer_ai* in conjunction with *engine_ai* will give you valuable insight into things to consider. What we wanted to
share here are some elements that are pretty key to success but that might not be directly evident at first.

## Speed is Also Key:

This may seem obvious, but the speed is key, and what we mean here is the latency of inference, which we measure here 
in the frames-per-second (FPS). When you use *engine_ai*, (https://github.com/miniautonomous/engine_ai), three FPS 
readings are presented in the UI: the primary drive loop, the inference rate, and the camera loop. This last FPS is 
generally an FYI that might help you debug if there seems to be a camera lag. The inference rate is how quickly your 
network can process frames, which in turn throttles your drive loop, which measure how quickly you go from camera image 
grab to steering/throttle output to the microcontroller. 

Generally real time is consider 30 FPS, and if you can achieve this when you are in an inference state, (i.e. 
autonomous), you are golden; that is, as long as your network actually works. The longer it takes the network to process
an image, the slower your inference rate and thus the slower your end-to-end drive loop. We have found from experience 
that regardless of the task at hand, 12 FPS is pretty much the bottom speed at which the vehicle can operate 
autonomously. The issue is that regardless how mundane the task, below 12 FBS, by the time the car has processed an 
image correctly, it is too far gone along a given trajectory to correctly process the current state of the vehicle's 
surroundings. (Thus beware of any demos when all you see is a vehicle go straight! ;D)

Why is this important to point out? Well, for some tasks, it is critical to add some memory to the model so that the 
network can 'anticipate' the driving condition it is in and help it maneuver. Memory is usually needed when there are no
key image markers, (things like lane tape on the floor or little cones, etc.). When you add memory, almost invariably 
your compute latency goes up. LSTM's and GRU's, (eventhough the latter are slightly less so), are computationally 
expensive. So when you train a model and the simulation you run on test files seems to show you that you have an 
accurate model, bear in mind that there is still a second metric that you need to determine: how fast will it run on the
car? Once you train the model, (and please always parse it to *TensorRT*), run it on Nano so that you see what are the 
resulting FPS of the drive and inference loops. If you are at or slightly above 15 FPS, normally you are at the edge of
reliable performance. By the time you hit 20 to 22, you are usually good to go. (The demo in which one of us is walking
the vehicle around the kitchen island ran at 21 FPS end-to-end.)

You may be tempted to ditch the state memory and just run end-to-end inference. The garage loop demo in our portal ran 
at a whopping 27-28 FPS. This allowed the car to go really, (really), fast, but the image markers of cones are a huge 
assistance and the only response required is to steer inward to avoid the obstacle. The complexity of the task will 
require judgement as to if you can go with or without memory.

## Focus on Data Acquisition

An early gating factor in distinguishing between remarkable success and dismal failure is the quality of training data. 
You may think that driving the vehicle around is pretty straightforward, and it is, but one needs to be mindful of a 
few key aspects. First, consistency; not so much in terms of driving behaviour, since some diversity is helpful to create
robustness, but in terms of lighting. Training a network to perform a task outside is very tricky since the incidental 
lighting varies so much during the day. If you train a great model at 10 a.m., and then run an inference drive test at 
noon, don't be surprised to face abject failure. If you are in an enclosed area, and you train with a closed loop and do
5 clockwise loops but only 2 counter-clockwise loops, don't be surprised if your network can only negotiate the track in 
one direction. This may seem obvious, but consider these factors as you train your model. Try to be patient at this 
stage of the cycle of training, since good quality data is really fundamental to your network's success.

## Dream the dream

This is not meant to be a commercial endorsement, but the Jetson Nano is really a very remarkable little piece of 
equipment. Currently, it consists of 128 GPU cores of the Maxwell variety, which is quite a punch for such a small 
footprint. The painful part is getting things compiled for its hardware stack, but once we got things set, we've been 
able to run things on the little platform that compare quite favorable to work we did 3 or 4 years ago on discrete GPU 
cards. Bottom line, if you can think of an autonomous task and you have doubts as to if the platform can handle it, try 
it out! You never know if a little ingenuity can help you get over the finish line and complete the task at hand!



