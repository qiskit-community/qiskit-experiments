---
title: 'Qiskit Experiments: A Python package to characterize and calibrate quantum computers'
tags:
  - Python
  - Quantum computing
  - Characterization
  - Calibration
authors:
  - name: Naoki Kanazawa
    orcid: 0000-0002-4192-5558
    affiliation: 1
  - name: Daniel J. Egger
    orcid: 0000-0002-5523-9807
    corresponding: true
    affiliation: 2
  - name: Yael Ben-Haim
    affiliation: 3
  - name: Helena Zhang
    orcid: 0000-0002-7813-7133
    affiliation: 4
  - name: Will E. Shanks
    orcid: 0000-0002-5045-8808
    affiliation: 4
  - name: Gadi Aleksandrowicz
    affiliation: 3
  - name: Christopher J. Wood
    orcid: 0000-0001-7606-7349
    affiliation: 4
affiliations:
 - name: IBM Quantum – IBM Research Tokyo, Tokyo, 103-8510, Japan
   index: 1
 - name: IBM Quantum, IBM Research Europe - Zurich, Ruschlikon 8003, Switzerland
   index: 2
 - name: IBM Quantum, IBM Research Israel, Haifa 31905, Israel
   index: 3
 - name: IBM Quantum, IBM T.J. Watson Research Center, Yorktown Heights, NY 10598, USA
   index: 4
date: 15 February 2023
bibliography: paper.bib

---

# Summary

Qiskit Experiments is a Python package for designing and running quantum computing experiments 
with a focus on calibration and characterization of quantum devices. 
It consists of a general purpose experiments framework which can be used by researchers to rapidly 
implement new experiments and a library of common experiments for calibration, characterization, 
and verification of quantum devices.

The core framework of `Qiskit Experiments` consists of three parts.
(i) An experiment class defines the quantum circuits to run. 
(ii) A data container class named `ExperimentData` stores the data measured during the execution 
of the quantum circuits.
(iii) An analysis class, attached to each experiment, defines how to analyze the measured data.
The analysis also attaches its results, such as fit results and figures, to the data container.
\autoref{fig:framework} summarizes this framework.
Importantly, this framework can interface with services to store, load, and share data.
The library of experiments includes common quantum computing experiments such as Randomized 
Benchmarking [@Magesan2011], Quantum State and Process Tomography [@Banaszek1999], Quantum Volume [@Cross2019], 
and gate error amplifying calibration sequences [@Tornow2022].

`Qiskit Experiments` is based on Qiskit [@Qiskit], a general purpose Python library for programming 
quantum computers and simulators.
It uses many other open source packages.
These include Numpy [@Harris2020] for fast numerical computing, lmfit [@Newville2014] to fit complex 
models to data, CVXPY [@Diamond2016] for convex optimization, Matplotlib [@Hunter2007] for plotting, 
and uncertainties [@Lebigot2016] to provide measurements with a mean and a standard deviation.


# Statement of need

Quantum computing processes information following the laws of quantum mechanics.
Quantum computers, like classical computers, must be programmed to perform quantum computations. 
A quantum computer consists of qubits which store information in 
quantum states, along with additional hardware elements, such as resonators to couple, control, 
and readout the qubits. 
The different elements in the quantum hardware have properties that must be characterized to 
calibrate the quantum gates that process the information. 
Furthermore, the quality of these gates has to be benchmarked to measure the overall performance 
of the quantum computer. 
This characterization and calibration requires an extensive set of experiments and analysis routines.

Quantum development software packages such as Qiskit, ReCirq [@QuantumAI], tKet [@Sivarajah2021], and
Forest [@Smith2016] are part of the quantum stack to execute quantum circuits on hardware.
They also enable high-level applications that abstract away the quantum hardware. 
Forest-benchmarking [@Forest] and pyGSTi [@pyGSTi] are tailored towards benchmarking of quantum hardware.
However, there is still a need for open-source software that enables researchers and hardware 
maintainers to easily execute characterization and calibration experiments.
`Qiskit Experiments` is unique in this perspective as it provides low-level characterization 
experiments that integrate with pulse-level control [@Alexander2020].
In addition, `Qiskit Experiments` provides a calibration framework to manage device calibration.
This framework is usable with any hardware exposed as a Qiskit backend.
For example, the `Qiskit Experiments` framework is used to explore measurements without qubit 
reset [@Tornow2022], benchmarking [@Amico2023], characterize positive operator value measures [@Fischer2022], quantum 
states [@Hamilton2022], and time-evolutions [@Greenaway2022], as well as calibrate gates [@Vazquez2022].

![
Conceptual framework of Qiskit Experiments.
The circuits are run as jobs on the quantum backends.
If an experiment exceeds the maximum circuit limit per job it is broken down in multiple jobs.
The raw data, figures and analysis results are contained in the `ExperimentData` class.
\label{fig:framework}](framework.pdf){ width=50% }

# Example usage

Here, we exemplify `Qiskit Experiments` with a Quantum Volume (QV) measurement [@Cross2019].
We execute random SU(4) circuits on a noisy simulator of a quantum backend to quantify
the largest quantum circuit with equal width and depth that can be successfully run.
A depth $d$ QV circuit is successful if it has mean heavy-output probability greater 
than two-thirds with a confidence level exceeding 0.977, and at least 100 trials have been run.
`Qiskit Experiments` only requires a few lines of code to run this standardized yet complex experiment.
The analysis classes of existing experiments automatically generate key figures with customizable 
visualization options, as exemplified by the QV plot in \autoref{fig:qv}.

![Example result of a quantum volume measurement carried out with `Qiskit Experiments`.
The dashed line shows the two-thirds threshold.
Each dot shows an execution of a randomized quantum circuit aggregated over many shots.
The shaded area is a $2\sigma$ confidence interval.
\label{fig:qv}](qv.pdf){ width=75% }

# Documentation

`Qiskit Experiments` documentation is available at [https://qiskit.org/documentation/experiments](https://qiskit.org/documentation/experiments).

# Acknowledgements

We acknowledge contributions from the Qiskit Community and feedback from our users.

# References
