# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ZZ Ramsey experiment
"""

from typing import List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from .analysis.zz_ramsey_analysis import ZZRamseyAnalysis


class ZZRamsey(BaseExperiment):
    r"""Experiment to characterize the static :math:`ZZ` interaction for a qubit pair

    # section: overview

        :math:`ZZ` can be expressed as the difference between the frequency of
        a qubit q_0 when another qubit q_1 is excited and the frequency of q_0
        when q_1 is in the ground state. Because :math:`ZZ` is symmetric in
        qubit index, it can also be expressed with the roles of q_0 and q_1
        reversed.  Experimentally, we measure :math:`ZZ` by performing Ramsey
        sequences on q_0 with q_1 in the ground state and again with q_1 in the
        excited state.

        Because we are interested in the difference in frequency between the
        two q_1 preparations rather than the absolute frequencies of q_0 for
        those preparations, we modify the Ramsey sequences (the circuits for
        the modified sequences are shown below). First, we add an X gate on q_0
        to the middle of the Ramsey delay. This would have the effect of
        echoing out the phase accumulation of q_0 (like a Hahn echo sequence),
        but we add a simultaneous X gate to q_1 as well. Flipping q_1 inverts
        the sign of the :math:`ZZ` term. The net result is that q_0 continues
        to accumulate phase proportional to :math:`ZZ` while the phase due to
        any ZI term is canceled out. This technique allows :math:`ZZ` to be
        measured using longer delay times that might otherwise be possible with
        a qubit with a slow frequency drift (i.e. the measurement is not
        sensitive to qubit frequency drift from shot to shot, only to drift
        within a single shot).

        The resulting q_0 state versus delay time data exhibit slow sinusoidal
        oscillations (assuming :math:`ZZ` is relatively small). To help with
        distinguishing between qubit decay and a slow oscillation, an extra Z
        rotation is applied before the final pulse on q_0. The angle of this Z
        rotation set proportional to the delay time of the sequence so that it
        acts like an extra rotation frequency common to the two q_1
        preparations. By looking at the difference in frequency fitted for the
        two cases, this common "fake" frequency (called ``f`` in the circuits
        shown below) is removed, leaving only the :math:`ZZ` value. The value
        of ``f`` in terms of the experiment options is ``zz_rotations /
        (max(delays) - min(delays))``.

        This experiment consists of following two circuits. The frequenc f is
        chosen based on the zz_rotations experiment option and the maximum
        delay time.

        .. parsed-literal::

            Modified Ramsey sequence with q_1 initially in the ground state

                 ┌────┐ ░ ┌─────────────────┐ ░ ┌───┐ ░ ┌─────────────────┐ ░ »
            q_0: ┤ √X ├─░─┤ Delay(delay[s]) ├─░─┤ X ├─░─┤ Delay(delay[s]) ├─░─»
                 └────┘ ░ └─────────────────┘ ░ ├───┤ ░ └─────────────────┘ ░ »
            q_1: ───────░─────────────────────░─┤ X ├─░─────────────────────░─»
                        ░                     ░ └───┘ ░                     ░ »
            c: 1/═════════════════════════════════════════════════════════════»
                                                                              »
            «     ┌─────────────────────┐┌────┐ ░ ┌─┐
            «q_0: ┤ Rz(4*delay*dt*f*pi) ├┤ √X ├─░─┤M├
            «     └────────┬───┬────────┘└────┘ ░ └╥┘
            «q_1: ─────────┤ X ├────────────────░──╫─
            «              └───┘                ░  ║
            «c: 1/═════════════════════════════════╩═
            «                                      0

            Modified Ramsey sequence with q_1 initially in the excited state

                 ┌────┐ ░ ┌─────────────────┐ ░ ┌───┐ ░ ┌─────────────────┐ ░ »
            q_0: ┤ √X ├─░─┤ Delay(delay[s]) ├─░─┤ X ├─░─┤ Delay(delay[s]) ├─░─»
                 ├───┬┘ ░ └─────────────────┘ ░ ├───┤ ░ └─────────────────┘ ░ »
            q_1: ┤ X ├──░─────────────────────░─┤ X ├─░─────────────────────░─»
                 └───┘  ░                     ░ └───┘ ░                     ░ »
            c: 1/═════════════════════════════════════════════════════════════»
                                                                              »
            «     ┌─────────────────────┐┌────┐ ░ ┌─┐
            «q_0: ┤ Rz(4*delay*dt*f*pi) ├┤ √X ├─░─┤M├
            «     └─────────────────────┘└────┘ ░ └╥┘
            «q_1: ──────────────────────────────░──╫─
            «                                   ░  ║
            «c: 1/═════════════════════════════════╩═
            «                                      0

    # section: analysis_ref

        :py:class:`ZZRamseyAnalysis`
    """

    def __init__(
        self,
        qubit: (int, int),
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            qubit: The qubits on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            experiment_options: experiment options to set
        """
        super().__init__(qubits=qubit, analysis=ZZRamseyAnalysis(), backend=backend)
        # Override the default of get_processor() which is "1" * num_qubits. We
        # only fit the probability of the target qubit.
        self.analysis.set_options(outcome="1")
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the :math:`ZZ` Ramsey experiment.

        Experiment Options:
            delays (list[float]): The list of delays that will be scanned in
                the experiment, in seconds. If not set, then ``num_delays``
                evenly spaced delays between ``min_delay`` and ``max_delay``
                are used. If ``delays`` is set, ``max_delay``, ``min_delay``,
                and ``num_delays`` are ignored.
            max_delay (float): Maximum delay time to use
            min_delay (float): Minimum delay time to use
            num_delays (int): Number of circuits to use per control state
                preparation
            zz_rotations (float): Number of full rotations of the Bloch vector
                if :math:`ZZ` is zero.
        """
        options = super()._default_experiment_options()
        options.delays = None
        options.min_delay = 0e-6
        options.max_delay = 10e-6
        options.num_delays = 50
        options.zz_rotations = 5

        return options

    @property
    def delays(self):
        """Delay values to use in circuits"""
        # This method allows delays to be set by the user as an explicit
        # sequence or as a minimum, maximum and number of values for a linearly
        # spaced sequence.
        options = self.experiment_options
        if options.delays is not None:
            return options.delays

        return np.linspace(options.min_delay, options.max_delay, options.num_delays)

    def _parameterized_circuits(
        self, delay_unit: str = "s"
    ) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Circuits for series 0 and 1 with several parameters

        Circuits are generated with parameters for:

            pi: the mathematical constant
            f: the synthetic frequency added onto the circuit
            dt: the conversion factor from the units used in the circuits'
                delay instructions to time in seconds.
            delay: the length of each delay instruction in each circuit

        The circuits are parameterized with these parameters so that
        ``QuantumCircuit.draw()`` produces a nice representation of the
        parameters in the circuit.

        Args:
            delay_unit: the unit of circuit delay instructions.

        Returns:
            Circuits for series 0 and 1
        """
        metadata = {
            "experiment_type": self._type,
            "qubits": self.physical_qubits,
            "unit": "s",
        }

        delay = Parameter("delay")
        freq = Parameter("f")
        dt = Parameter("dt")
        pi = Parameter("pi")

        delay_time_total = 2 * delay * dt

        # Template circuit for series 0
        # Control qubit starting in |0> state, flipping to |1> in middle
        circ0 = QuantumCircuit(2, 1, metadata=metadata.copy())
        circ0.metadata["series"] = "0"

        circ0.sx(0)

        circ0.barrier()
        circ0.delay(delay, 0, delay_unit)
        circ0.barrier()

        circ0.x(0)
        circ0.x(1)

        circ0.barrier()
        circ0.delay(delay, 0, delay_unit)
        circ0.barrier()

        circ0.rz(2 * pi * freq * delay_time_total, 0)
        circ0.sx(0)
        # Flip control back to 0, so control qubit is in 0 for all circuits
        # when qubit 1 is measured.
        circ0.x(1)

        circ0.barrier()
        circ0.measure(0, 0)

        # Template circuit for series 1
        # Control qubit starting in |1> state, flipping to |0> in middle
        circ1 = QuantumCircuit(2, 1, metadata=metadata.copy())
        circ1.metadata["series"] = "1"

        circ1.x(1)
        circ1.sx(0)

        circ1.barrier()
        circ1.delay(delay, 0, delay_unit)
        circ1.barrier()

        circ1.x(0)
        circ1.x(1)

        circ1.barrier()
        circ1.delay(delay, 0, delay_unit)
        circ1.barrier()

        circ1.rz(2 * pi * freq * delay_time_total, 0)
        circ1.sx(0)

        circ1.barrier()
        circ1.measure(0, 0)

        return circ0, circ1

    def _template_circuits(
        self, dt_value: float = 1.0, delay_unit: str = "s"
    ) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Template circuits for series 0 and 1 parameterized by delay

        The generated circuits have the length of the delay instructions as the
        only parameter.

        Args:
            dt_value: the value of the backend ``dt`` value. Used to convert
                delay values into units of seconds.
            delay_unit: the unit of circuit delay instructions.

        Returns:
            Circuits for series 0 and 1
        """
        circ0, circ1 = self._parameterized_circuits(delay_unit=delay_unit)

        # Simulated frequency applied to both sets of circuits. The value is
        # chosen to induce zz_rotations number of rotation within the time
        # window that the delay is swept through.
        options = self.experiment_options
        common_freq = options.zz_rotations / (max(self.delays) - min(self.delays))

        def get(circ, param):
            return next(p for p in circ.parameters if p.name == param)

        for circ in (circ0, circ1):
            assignments = {
                get(circ, "pi"): np.pi,
                get(circ, "f"): common_freq,
                get(circ, "dt"): dt_value,
            }
            circ.assign_parameters(assignments, inplace=True)

        return circ0, circ1

    def circuits(self) -> List[QuantumCircuit]:
        """Create circuits for :math:`ZZ` Ramsey experiment

        Returns:
            A list of circuits with a variable delay.
        """
        timing = BackendTiming(self.backend)
        if timing.dt is not None:
            dt_val = timing.dt
        else:
            # If the backend does not have dt, we treat it as 1.0 when it is
            # used the conversion factor to seconds because the delays are
            # always handled in seconds in this case.
            dt_val = 1.0

        circ0, circ1 = self._template_circuits(dt_value=dt_val, delay_unit=timing.delay_unit)
        circs = []

        for delay in self.delays:
            for circ in (circ0, circ1):
                assigned = circ.assign_parameters(
                    {circ.parameters[0]: timing.round_delay(time=delay / 2)}, inplace=False
                )
                assigned.metadata["xval"] = 2 * timing.delay_time(time=delay / 2)
                circs.append(assigned)

        return circs
