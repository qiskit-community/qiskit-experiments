What is Qiskit Experiments?
===========================

Qiskit Experiments is a package for running device characterization and calibration 
experiments on top of Qiskit Terra.

An **experiment**
is a series of circuits, executed on a device, and analysis of
of the subsequent data that's all built on top of the core functionality
of Qiskit Terra. Once an experiment is executed on a quantum backend through a series 
of jobs, analysis is run automatically and results in the form of data and figures are generated.

In addition to the experiment framework itself, Qiskit Experiments also has a rich 
library of experiments for calibrating and characterizing qubits.

What Qiskit Experiments can do
==============================

* Run characterization and calibration experiments such as quantum
  volume and randomized benchmarking
* Run built-in or customized experiments with all the options that Terra has
* Specify fit series and parameters in the analysis
* Transform the data through the data processor
* Flexible visualization, storage, and retrieval of data

Qiskit Experiments is for
* Experimentalists who want to characterize and calibrate devices
* 

A quick primer
==============

The Qiskit Experiments package consists of the experimental framework, 

.. figure:: ./images/experimentarch.png
    :width: 400
    :align: center


Experiments start with an ``Experiment`` class, which instantiates the circuits that
will be run and also the metadata and options that will be used for the experiment, 
transpilation, execution, and analysis. During execution, circuits are automatically
packaged into one or more jobs for the specified backend device.

Each ``Experiment`` class is tied to its corresponding ``Analysis`` class. Once jobs
complete execution, the ``Analysis`` class processes and analyzes raw data to output 
an ``ExperimentData`` class that contains
the resulting analysis results, figures, metadata, as well as the original raw data.

