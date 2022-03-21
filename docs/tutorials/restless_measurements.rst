Restless Measurements
=====================

When running circuits the qubits are typically reset to the ground state after
each measurement to ensure that the next circuit has a well-defined initial state.
This can be done passively by waiting several :math:`T_1`-times so that qubits in
the excited state decay to :math:`\left\vert0\right\rangle`. Since :math:`T_1`-times
are continuously improving (they already increased beyond :math:`100\,\mu s`), this
initialization procedure is now inefficient. This makes active reset necessary.
Active qubit reset, as employed by IBM Quantum systems is more efficient and saves
time but also lasts a few microseconds (between :math:`3` and :math:`5\,\mu s`).
Furthermore, a delay, typically lasting :math:`250\,\mu s`, after the reset
operation is often necessary to ensure a high initialization quality.
However, for several types of characterization and calibration experiments qubit
reset is not needed and we can directly continue with the subsequent circuit after
a short delay, even if the qubit was measured in the excited state. Foregoing qubit
reset is the main idea behind restless measurements.

The IBM Quantum devices have dynamical repetition delays enabled. This means that
we can choose the delay between the execution of two quantum circuits. This delay
can typically range from :math:`0` to :math:`500\,\mu s`. The default value for
most of the devices is :math:`250\,\mu s`. Restless measurements set this delay to
a small value such as :math:`1\,\mu s`.

When the qubit is not reset it will either be in the :math:`\left\vert0\right\rangle`
or in the :math:`\left\vert1\right\rangle` state when the next circuit starts.
Therefore, the measured outcomes of the restless experiments require post-processing.
The following example, taken from Ref. [1], illustrates what happens to the single
measurement outcomes represented as complex numbers in the IQ plane in a restless
setting. Here, we run three circuits with an identity gate and three circuits with
an X gate, each followed by a measurement. The numbers in the IQ shots indicate the
order in which the shots were acquired. The IQ plane on the left shows the single
measurement shots gathered when the qubits are reset. Here, the blue and red points
corresponding to measurements following the Id and X gates are associated with the
:math:`\left\vert0\right\rangle` and :math:`\left\vert1\right\rangle` states,
respectively.
By contrast, with restless measurements the qubit is not reset after a
measurement. As one can see in the IQ plane on the left the single measurement
outcomes of the Id and X circuits no longer match with the
:math:`\left\vert0\right\rangle` and :math:`\left\vert1\right\rangle` states,
respectively. This is why restless measurements need special post-processing.

.. image:: restless_shots.png
   :width: 600

Enabling restless measurements
------------------------------

In Qiskit Experiments the experiments that support restless measurements
inherit from the ``RestlessMixin``. This mix-in class adds methods to set
restless run options and defines the data processor that will process the
measured data. Here, we will show how to activate restless measurements using
a fake backend and a rough Drag experiment. Note however, that you will not
be able to run the experiment since only real backends support restless
measurements.

.. jupyter-execute::

    from qiskit_experiments.library import RoughDragCal
    from qiskit_experiments.calibration_management import (
        Calibrations,
        FixedFrequencyTransmon,
    )
    from qiskit_experiments.data_processing.data_processor import DataProcessor
    from qiskit.test.mock import FakeBogota

    # replace this lines with an IBM Quantum backend to run the experiment.
    backend = FakeBogota()
    cals = Calibrations.from_backend(backend, library=FixedFrequencyTransmon())

    # Define the experiment
    qubit = 2
    cal_drag = RoughDragCal(qubit, cals, schedule_name='sx', backend=backend)

    # Enable restless measurements by setting the run options and data processor
    cal_drag.enable_restless(rep_delay=1e-6)

After calling ``enable_restless`` the experiment is ready to be run in a restless
mode. With a hardware backend this would be done by calling the ``run`` method

.. code:: python

    drag_data_restless = cal_drag.run()

As shown by the example, the code is identical to running a normal
experiment aside from a call to the method ``enable_restless``. This method
will set the data processor that post-processes the restless measured shots
according to the order in which they were acquired. You can also chose
to keep the standard data processor by providing it to the analysis
options and telling ``enable_restless`` not to override the data processor

.. jupyter-execute::

    from qiskit_experiments.data_processing import (
        DataProcessor,
        Probability,
    )

    # define a standard data processor.
    standard_processor = DataProcessor("counts", [Probability("1")])

    cal_drag = RoughDragCal(qubit, cals, schedule_name='sx', backend=backend)
    cal_drag.set_analysis_options(data_processor=standard_processor)

    # enable restless mode and set override_processor_by_restless to False.
    cal_drag.enable_restless(rep_delay=1e-6, override_processor_by_restless=False)

If you run the experiment in this setting you will see that the data is often
unusable which illustrates the importance of the data processing. As detailed
in Ref. [2] restless measurements can be done with a wide variety
of experiments such as fine amplitude and drag error amplifying gate sequences
as well as randomized benchmarking.

References
~~~~~~~~~~

[1] Max Werninghaus, Daniel J. Egger, Stefan Filipp, High-speed calibration and
characterization of superconducting quantum processors without qubit reset,
PRX Quantum 2, 020324 (2021).

[2] Caroline Tornow, Naoki Kanazawa, William E. Shanks, Daniel J. Egger,
Minimum quantum run-time characterization and calibration via restless
measurements with dynamic repetition rates,
https://arxiv.org/abs/2202.06981

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright