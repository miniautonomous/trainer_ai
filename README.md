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

3. [Simulation is the Key to Success](#simulation-is-the-key-to-success)

3. [Training End-to-End Networks](#training-end-to-end-networks)

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

The focal script is *trainer_ai.py*

# Simulation is the Key to Success

# Training End-to-End Networks