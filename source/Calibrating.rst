Calibrating single-qubit gates on a pulse simulation backend
=================================================================

In this tutorial we demonstrate how to calibrate single-qubit gates 
on a pulse backend using the calibration framework in qiskit-experiments. 
We will run experiments to find the qubit frequency, 
calibrate the amplitude of DRAG pulses 
and chose the value of the DRAG parameter that minimizes leakage. 
The calibration framework requires the user to
- setup an instance of `Calibrations`,
- run calibration experiments which can be found in `qiskit_experiment.library.calibration`.

Note that the values of the parameters stored in the instance of the 
Calibrations class will automatically be updated by the calibration experiments. 
This automatic updating can also be disabled using the `auto_update` flag.

.. jupyter-execute::

    import pandas as pd
    import numpy as np
    import qiskit.pulse as pulse
    from qiskit.circuit import Parameter
    from qiskit_experiments.calibration_management.calibrations import Calibrations
    from qiskit import schedule
    from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend

.. jupyter-execute::

    backend = SingleTransmonTestBackend()
    qubit = 0