AC Stark effect
===============

When a qubit is driven with an off-resonant tone,
the qubit frequency :math:`f_0` is slightly shifted through what is known as the (AC) Stark effect.
This technique is sometime used to characterize qubit properties in the vicinity of
the base frequency, especially with the fixed frequency qubit architecture which usually
doesn't have a knob to control frequency [1].

The important control parameters of the Stark effect is the amplitude
:math:`\Omega` and frequency :math:`f_S` of
the off-resonant tone, which we will call *Stark tone* in the following.
In the low power limit, the amount of frequency shift :math:`\delta f_S`
that the qubit may experience is described as follows [2]:

.. math::

    \delta f_S \propto \frac{\alpha}{2\Delta\left(\alpha + \Delta\right)} \Omega^2,

where :math:`\alpha` is the qubit anharmonicity and :math:`\Delta=f_S - f_0` is the
frequency separation of the Stark tone from the qubit frequency :math:`f_0`.
We sometime call this *Stark shift*.
As you can see in the equation above, :math:`\Delta=0` yields a singular point
where :math:`\delta f_S` diverges. This corresponds to the Rabi drive,
where the qubit is driven on resonance.
This indicates the collision of the Stark tone frequency, and thus the tone must be
well separated in the spectrum from the qubit frequency
or any other transition frequencies such as
the two-photon transition at :math:`f_0 + \alpha/2`.


.. _stark_tone_implementation:

Stark tone implementation in Qiskit
-----------------------------------

Usually, we fix the Stark tone frequency :math:`f_S` and control the amplitude :math:`\Omega`
to modulate qubit frequency.
In Qiskit, we often use an abstracted amplitude :math:`\bar{\Omega}`,
instead of the physical amplitude :math:`\Omega` in the experiments.

Because the Stark shift :math:`\delta f_S` has a quadratic dependence on
the tone amplitude :math:`\Omega`, the resulting shift is not sensitive to its sign.
On the other hand, the sign of the shift depends on the sign of the frequency offset :math:`\Delta`.
In a typical parameter regime of :math:`|\Delta | < | \alpha |`,

.. math::

    \text{sign}(\delta f_S) = - \text{sign}(\Delta).

In other words, positive (negative) Stark shift occurs when the tone frequency :math:`f_S`
is lower (higher) than the qubit frequency :math:`f_0`.
When an experimentalist wants to perform spectroscopy of some qubit parameter
in the vicinity of :math:`f_0`, one must manage the sign of :math:`f_S`
in addition to the magnitude of :math:`\Omega` as they need to
switch the sign of the Stark shift.

To alleviate such experimental complexity, the abstracted amplitude :math:`\bar{\Omega}`
with virtual sign is introduced in Qiskit Experiments:

.. math::

    \Delta &= - \text{sign}(\bar{\Omega}) | \Delta |, \\
    \Omega &= | \bar{\Omega} |.

Stark experiments in Qiskit usually take two control parameters :math:`(\bar{\Omega}, |\Delta|)`,
usually specified by ``stark_amp`` and ``stark_freq_offset`` in the experiment options, respectively.
In this representation, the sign of the Stark shift matches the sign of :math:`\bar{\Omega}`.

.. math::

    \text{sign}(\delta f_S) = \text{sign}(\bar{\Omega})

This allows an experimentalist to control both sign and amount of
the Stark shift with the ``stark_amp``. In the superconducting qubit setup,
the ``stark_freq_offset`` is a fixed positive value in :math:`0 \ll |\Delta| < |\alpha/2|`.
In reality, this condition might be more complicated due to transition levels of the
nearest neighbor qubits, and it must be carefully chosen to avoid frequency collisions [3].


.. _stark_channel_consideration:

Stark tone channel
------------------

It may be necessary to supply a pulse channel to apply the Stark tone.
In Qiskit Experiments, the Stark experiments usually have an experiment option ``stark_channel``
to specify this.
By default, the Stark tone is applied to the same channel as the qubit drive
with a frequency shift. This frequency shift might update the channel frame,
which accumulates unwanted phase against the frequency difference between
the qubit drive :math:`f_0` and Stark tone frequencies :math:`f_S` in addition to
the qubit Stark shfit :math:`\delta f_s`.
You can use a dedicated Stark drive channel if available.
Otherwise, you may want to use a control channel associated with the physical
drive port of the qubit.

In a typical IBM device of the cross-resonance drive architecture,
such channel can be identified with your backend.

.. jupyter-execute::

    from qiskit.providers.fake_provider import FakeHanoi

    qubit = 0
    coupling_map = backend.configuration().coupling_map

    for qpair in coupling_map:
        if qpair[0] == qubit:
            break

    print(backend.configuration().control(qpair)[0])

This returns a control channel for which the qubit is the control qubit.
This depends on the architecture of your quantum device.


References
----------

[1] Malcolm Carroll, Sami Rosenblatt, Petar Jurcevic, Isaac Lauer and Abhinav Kandala,
Dynamics of superconducting qubit relaxation times, npj Quantum Inf 8, 132 (2022).
https://arxiv.org/abs/2105.15201

[2] Easwar Magesan, Jay M. Gambetta, Effective Hamiltonian models of the cross-resonance gate,
Phys. Rev. A 101, 052308 (2020).
https://arxiv.org/abs/1804.04073

[3] Jared B. Hertzberg, Eric J. Zhang, Sami Rosenblatt, et. al.,
Laser-annealing Josephson junctions for yielding scaled-up superconducting quantum processors,
npj Quantum Information 7, 129 (2021).
https://arxiv.org/abs/2009.00781



.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
