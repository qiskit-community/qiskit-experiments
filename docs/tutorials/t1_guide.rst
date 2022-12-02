A :math:`T_1` experiment
========================

In a :math:`T_1` experiment, we measure an excited qubit after a delay.
Due to decoherence processes (e.g. amplitude damping channel), it is
possible that, at the time of measurement, after the delay, the qubit
will not be excited anymore. The larger the delay time is, the more
likely is the qubit to fall to the ground state. The goal of the
experiment is to characterize the decay rate of the qubit towards the
ground state.

We start by fixing a delay time :math:`t` and a number of shots
:math:`s`. Then, by repeating :math:`s` times the procedure of exciting
the qubit, waiting, and measuring, we estimate the probability to
measure :math:`|1\rangle` after the delay. We repeat this process for a
set of delay times, resulting in a set of probability estimates.

In the absence of state preparation and measurement errors, the
probability to measure \|1> after time :math:`t` is :math:`e^{-t/T_1}`,
for a constant :math:`T_1` (the coherence time), which is our target
number. Since state preparation and measurement errors do exist, the
qubit’s decay towards the ground state assumes the form
:math:`Ae^{-t/T_1} + B`, for parameters :math:`A, T_1`, and :math:`B`,
which we deduce form the probability estimates. To this end, the
:math:`T_1` experiment internally calls the ``curve_fit`` method of
``scipy.optimize``.


.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
