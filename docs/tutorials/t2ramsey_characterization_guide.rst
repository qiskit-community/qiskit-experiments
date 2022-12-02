T2 Ramsey Characterization
==========================

The purpose of the :math:`T_2`\ Ramsey experiment is to determine two of
the qubit’s properties: *Ramsey* or *detuning frequency* and
:math:`T_2^\ast`. The rough frequency of the qubit was already
determined previously. The control pulses are based on this frequency.

In this experiment, we would like to get a more precise estimate of the
qubit’s frequency. The difference between the frequency used for the
control rotation pulses, and the precise frequency is called the
*detuning frequency*. This part of the experiment is called a *Ramsey
Experiment*. :math:`T_2^\ast` represents the rate of decay toward a
mixed state, when the qubit is initialized to the
:math:`\left|1\right\rangle` state.

Since the detuning frequency is relatively small, we add a phase gate to
the circuit to enable better measurement. The actual frequency measured
is the sum of the detuning frequency and the user induced *oscillation
frequency* (``osc_freq`` parameter).

The circuit used for the experiment comprises the following:

::

   1. Hadamard gate
   2. delay
   3. RZ gate that rotates the qubit in the x-y plane 
   4. Hadamard gate
   5. measurement

The user provides as input a series of delays (in seconds) and the
oscillation frequency (in Hz). During the delay, we expect the qubit to
precess about the z-axis. If the p gate and the precession offset each
other perfectly, then the qubit will arrive at the
:math:`\left|0\right\rangle` state (after the second Hadamard gate). By
varying the extension of the delays, we get a series of oscillations of
the qubit state between the :math:`\left|0\right\rangle` and
:math:`\left|1\right\rangle` states. We can draw the graph of the
resulting function, and can analytically extract the desired values.

The resulting graph of this experiment has the form:
:math:`f(t) = a^{-t/T_2*} \cdot \cos(2 \pi f t + \phi) + b` where *t* is
the delay, :math:`T_2^\ast` is the decay factor, and *f* is the detuning
frequency.


Providing initial user estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can provide initial estimates for the parameters to help the
analysis process. Because the curve is expected to decay toward
:math:`0.5,` the natural choice for parameters :math:`A` and :math:`B`
is :math:`0.5`. Varying the value of :math:`\phi` will shift the graph
along the x-axis. Since this is not of interest to us, we can safely
initialize :math:`\phi` to 0. In this experiment, ``t2ramsey`` and ``f``
are the parameters of interest. Good initial estimates for them are values
computed in previous experiments on this qubit or a similar values
computed for other qubits.

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
