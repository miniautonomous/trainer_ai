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

The key here is the content of **input_script.txt**, so lets review two examples: one script that trains a network
with state memory, (i.e. contains a GRU, LSTM, etc.), and one that trains a stateless model, (i.e. image in, inference
out). Let's start with the later first since it is the most basic.

## Stateless Model




## A Sample Network Definition File

# Simulation is the Key to Success

# Training End-to-End Networks