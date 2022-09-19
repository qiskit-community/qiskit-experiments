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

import math
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter

from qiskit_experiments.framework import BaseExperiment, Options
from .zz_ramsey_analysis import ZZRamseyAnalysis


class BackendTimingMixin:
    """Mixin class for ``BaseExperiment`` to provide timing helper methods

    The methods and properties provided by this class help with calculating
    delay and pulse timing that depends on the timing constraints of the
    backend. They abstract away the necessary accessing of the backend object.

    .. note::

        The methods in this class assume that the ``backend`` attribute is
        constant. Methods should not call methods of this class before and
        after modifying the ``backend`` attribute and expect consistent
        results.
    """

    @property
    def _delay_unit(self) -> str:
        """The delay unit for the current backend

        "dt" is used if dt is present in the backend configuration. Otherwise
        "s" is used.
        """
        if self.backend is not None and hasattr(self.backend.configuration(), "dt"):
            return "dt"

        return "s"

    @property
    def _dt(self) -> float:
        """Backend dt value

        Raises:
            QiskitError: The backend is not set or does not include a dt value.
        """
        if self.backend is not None and hasattr(self.backend.configuration(), "dt"):
            return self.backend.configuration().dt

        raise QiskitError("Backend must be set to consider sample timing.")

    def _delay_duration(self, time: float, next_instruction: str = "pulse") -> Union[int, float]:
        """The delay duration closest to ``time`` in delay units

        This method produces the value to pass for the ``duration`` of a
        ``Delay`` instruction of a ``QuantumCircuit`` so that the delay fills
        the time until the next pulse.  This method does a little more than
        just convert ``time`` to the closest number of samples. The pulse
        timing constraints of the backend are considered in order to give a
        number of samples closest to ``time`` plus however many more samples
        are needed to get to the next valid sample for the start of a pulse in
        a subsequent instruction.

        This method is useful when it is desired that a delay instruction
        accurately reflect how long the hardware will pause between two other
        instructions.

        Args:
            time: The nominal delay time to convert
            next_instruction: Either "pulse", "acquire", or "both" to indicate
                whether the next instruction will be a pulse, an acquire
                instruction, or both.

        Returns:
            The delay duration in samples or seconds, depending on the value of
            :meth:`.BackendTimingMixin._delay_unit`.

        Raises:
            QiskitError: Bad value for ``next_instruction``.
        """
        if self._delay_unit == "s":
            return time

        timing_constraints = getattr(self.backend.configuration(), "timing_constraints", {})
        pulse_alignment = timing_constraints.get("pulse_alignment", 1)
        acquire_alignment = timing_constraints.get("acquire_alignment", 1)

        if next_instruction == "both":
            # We round the delay values to the least common multiple of the
            # alignment constraints so that the delay can fill all the time
            # until a subsequent pulse satisfying the alignment constraints.
            #
            # Replace with math.lcm(pulse_alignment, acquire_alignment) when
            # dropping support for Python 3.8
            granularity = (
                pulse_alignment * acquire_alignment // math.gcd(pulse_alignment, acquire_alignment)
            )
        elif next_instruction == "pulse":
            granularity = pulse_alignment
        elif next_instruction == "acquire":
            granularity = acquire_alignment
        else:
            raise QiskitError(f"Bad value for next_instruction: {next_instruction}")

        samples = int(round(time / self._dt / granularity) * granularity)

        return samples

    def _pulse_duration(self, time: float) -> int:
        """The number of samples giving a valid pulse duration closest to ``time``

        Args:
            time: Pulse duration in seconds

        Returns:
            The number of samples corresponding to ``time``

        Raises:
            QiskitError: The backend timing constraints' min_length is not a
                multiple of granularity
        """
        timing_constraints = getattr(self.backend.configuration(), "timing_constraints", {})
        granularity = timing_constraints.get("granularity", 1)
        min_length = timing_constraints.get("min_length", 1)

        samples = int(round(time / self._dt / granularity)) * granularity
        samples = max(samples, min_length)

        if min_length % granularity != 0:
            raise QiskitError("Backend timing does not match assumptions!")

        return samples

    def _delay_time(self, time: float, next_instruction: str = "pulse") -> float:
        """The closest actual delay time in seconds to ``time``

        This method uses :meth:`.BackendTimingMixin._delay_duration` and then
        converts back into seconds.

        Args:
            time: The nominal delay time to be rounded

        Returns:
            The realizable delay time in seconds
        """
        if self._delay_unit == "s":
            return time

        return self._dt * self._delay_duration(time, next_instruction)

    def _pulse_time(self, time: float) -> float:
        """The closest hardware-realizable pulse duration to ``time`` in seconds

        This method uses :meth:`.BackendTimingMixin._pulse_duration` and then
        converts back into seconds.

        Args:
            time: The nominal pulse time to be rounded

        Returns:
            The realizable pulse time in seconds
        """
        return self._dt * self._n_pulse(time)


class ZZRamsey(BaseExperiment, BackendTimingMixin):
    r"""Experiment to characterize the static :math:`ZZ` interaction for a qubit pair

    # section: Overview

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

    @property
    def _instr_delay_to_time(self) -> float:
        """Conversion factor from delay instruction value to seconds"""
        if self._delay_unit == "dt":
            return self._dt

        return 1.0

    def _parameterized_circuits(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
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
        circ0.delay(delay, 0, self._delay_unit)
        circ0.barrier()

        circ0.x(0)
        circ0.x(1)

        circ0.barrier()
        circ0.delay(delay, 0, self._delay_unit)
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
        circ1.delay(delay, 0, self._delay_unit)
        circ1.barrier()

        circ1.x(0)
        circ1.x(1)

        circ1.barrier()
        circ1.delay(delay, 0, self._delay_unit)
        circ1.barrier()

        circ1.rz(2 * pi * freq * delay_time_total, 0)
        circ1.sx(0)

        circ1.barrier()
        circ1.measure(0, 0)

        return circ0, circ1

    def _template_circuits(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Template circuits for series 0 and 1 parameterized by delay

        The generated circuits have the length of the delay instructions as the
        only parameter.

        Returns:
            Circuits for series 0 and 1
        """
        circ0, circ1 = self._parameterized_circuits()

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
                get(circ, "dt"): self._instr_delay_to_time,
            }
            circ.assign_parameters(assignments, inplace=True)

        return circ0, circ1

    def circuits(self) -> List[QuantumCircuit]:
        """Create circuits for :math:`ZZ` Ramsey experiment

        Returns:
            A list of circuits with a variable delay.
        """
        circ0, circ1 = self._template_circuits()
        circs = []
        for delay in self.delays:
            for circ in (circ0, circ1):
                assigned = circ.assign_parameters(
                    {circ.parameters[0]: self._delay_duration(delay / 2, "pulse")}, inplace=False
                )
                assigned.metadata["xval"] = 2 * self._delay_time(delay / 2, "pulse")
                circs.append(assigned)

        return circs
