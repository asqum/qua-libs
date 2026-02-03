# AS winter school qua installation guide

This guide will help you install the QUA programming language and its associated tools on your local machine. After the installation, you will be ready to write QUA programs and send the execution jobs to OPX1000. Follow the steps below to get started.

## Step 1: create a  virtual environment
It is recommended to create a virtual environment to manage the dependencies for QUA. You can use `conda` for this purpose.

### Using conda
```bash
conda create -n qua-env python=3.11
conda activate qua-env
```

## Step 2: install QUA and related packages
First, activate your virtual environment if you haven't already:
```bash
conda activate qua-env
``` 
Then, install the QUA package along with other necessary libraries using pip:
```bash
pip install qm-qua, qualang_tools
```
# Basic files
You can find some example QUA programs in this folder. 

### Configuration
- [configuration.py](configuration.py): This file contains the configuration for the opx1000. You can modify it according to your hardware setup.

### QUA Syntax basics
- [00_hello_qua_1](00_hello_qua_1.py): It shows how we can play a xy, z and resonator pulse at the same time by using the QUA commands. 

- [00_hello_qua_2](00_hello_qua_2.py): It shows how we can use `align()` to order multiple elements in QUA.

- [00_hello_qua_3](00_hello_qua_3.py): It shows how we can use `wait()` to wait multiple elements in QUA.

- [00_hello_qua_4](00_hello_qua_4.py): It shows how we can dynamically sweep pulse duration in QUA.

- [00_hello_qua_5](00_hello_qua_5.py): It shows how we can dynamically sweep pulse amplitude in QUA.

### Hardware calibration
- [02_raw_adc_traces_mw_fem](02_raw_adc_traces_mw_fem.py): A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets and check the multiplexed readout levels.

- [03_time_of_flight_mw_fem](03_time_of_flight_mw_fem.py): A script to measure the ADC offsets and calibrate the time of flight.

- [Resonator Spectroscopy](04_resonator_spectroscopy_single.py) - Performs a 1D frequency sweep on a given resonator.

# Exercise
The goal of this exercise is to get familiar with the QUA programming language and the basic hardware calibration procedures. By completing the provided scripts, you will learn how to write QUA programs for the OPX1000, and perform essential calibration tasks for quantum experiments.

## Task1: T1 experiment
## Task2: Ramsey experiment