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

from typing import List, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter, ParameterExpression

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from .analysis.zz_ramsey_analysis import ZZRamseyAnalysis


class ZZRamsey(BaseExperiment):
    r"""An experiment to characterize the static :math:`ZZ` interaction for a qubit pair.

    # section: overview

        This experiment assumes a two qubit Hamiltonian of the form

        .. math::

            H = h \left(\frac{f_0}{2} ZI + \frac{f_1}{2} IZ + \frac{f_{ZZ}}{4} ZZ\right)

        and measures the strength :math:`f_{ZZ}` of the :math:`ZZ` term.
        :math:`f_{ZZ}` can be described as the difference between the frequency
        of qubit 0 when qubit 1 is excited and the frequency of qubit 0 when
        qubit 1 is in the ground state. Because :math:`f_{ZZ}` is symmetric in
        qubit index, it can also be expressed with the roles of 0 and 1
        reversed.  Experimentally, we measure :math:`f_{ZZ}` by performing
        Ramsey sequences on qubit 0 with qubit 1 in the ground state and again
        with qubit 1 in the excited state. The standard Ramsey experiment
        consists of putting a qubit along the :math:`X` axis of Bloch sphere,
        waiting for some time, and then measuring the qubit project along
        :math:`X`. By measuring the :math:`X` projection versus time the qubit
        frequency can be inferred. See
        :class:`~qiskit_experiments.library.characterization.T2Ramsey` and
        :class:`~qiskit_experiments.library.characterization.RamseyXY`.

        Because we are interested in the difference in qubit 0 frequency
        between the two qubit 1 preparations rather than the absolute
        frequencies of qubit 0 for those preparations, we modify the Ramsey
        sequences (the circuits for the modified sequences are shown below).
        First, we add an X gate on qubit 0 to the middle of the Ramsey delay.
        This would have the effect of echoing out the phase accumulation of
        qubit 0 (like a Hahn echo sequence as used in
        :class:`~qiskit_experiments.library.characterization.T2Hahn`), but we
        add a simultaneous X gate to qubit 1 as well.  Flipping qubit 1 inverts
        the sign of the :math:`f_{ZZ}` term. The net result is that qubit 0
        continues to accumulate phase proportional to :math:`f_{ZZ}` while the
        phase due to any ZI term is canceled out. This technique allows
        :math:`f_{ZZ}` to be measured using longer delay times than might
        otherwise be possible with a qubit with a slow frequency drift (i.e.
        the measurement is not sensitive to qubit frequency drift from shot to
        shot, only to drift within a single shot).

        The resulting excited state population of qubit 0 versus delay time
        exhibits slow sinusoidal oscillations (assuming :math:`f_{ZZ}` is
        relatively small). To help with distinguishing between qubit decay and
        a slow oscillation, an extra Z rotation is applied before the final
        pulse on qubit 0. The angle of this Z rotation is set proportional to
        the delay time of the sequence. This angle proportional to time behaves
        similarly to measuring at a fixed angle with the qubit rotating at a
        constant frequency. This virtual frequency is common to the two qubit 1
        preparations. By looking at the difference in frequency fitted for the
        two cases, this virtual frequency (called :math:`f` in the circuits
        shown below) is removed, leaving only the :math:`f_{ZZ}` value. The
        value of :math:`f` in terms of the experiment options is
        ``num_rotations / (max(delays) - min(delays))``.

        This experiment consists of the following two circuits repeated with
        different ``delay`` values.

        .. parsed-literal::

            Modified Ramsey sequence with qubit 1 initially in the ground state

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

            Modified Ramsey sequence with qubit 1 initially in the excited state

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

        :class:`ZZRamseyAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.zzramsey_test_backend import ZZRamseyTestBackend

            backend = ZZRamseyTestBackend(zz_frequency=50e3)

        .. jupyter-execute::

            from qiskit_experiments.library.characterization import ZZRamsey

            exp = ZZRamsey(physical_qubits=(0,1), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(
        self,
        physical_qubits: Tuple[int, int],
        backend: Union[Backend, None] = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            physical_qubits: The qubits on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            experiment_options: experiment options to set
        """
        super().__init__(physical_qubits, analysis=ZZRamseyAnalysis(), backend=backend)
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
            max_delay (float): Maximum delay time to use.
            min_delay (float): Minimum delay time to use.
            num_delays (int): Number of circuits to use per control state
                preparation.
            num_rotations (float): The extra rotation added to qubit 0 uses a
                frequency that gives this many rotations in the case where
                :math:`f_{ZZ}` is 0.
        """
        options = super()._default_experiment_options()
        options.delays = None
        options.min_delay = 0e-6
        options.max_delay = 10e-6
        options.num_delays = 50
        options.num_rotations = 5

        return options

    def delays(self) -> List[float]:
        """Delay values to use in circuits

        Returns:
            The list of delays to use for the different circuits based on the
            experiment options.
        """
        # This method allows delays to be set by the user as an explicit
        # sequence or as a minimum, maximum and number of values for a linearly
        # spaced sequence.
        options = self.experiment_options
        if options.delays is not None:
            return options.delays

        return np.linspace(options.min_delay, options.max_delay, options.num_delays).tolist()

    def frequency(self) -> float:
        """Frequency of qubit rotation when ZZ is 0

        This method calculates the simulated frequency applied to both sets of
        circuits. The value is chosen to induce `num_rotations` number of
        rotation within the time window that the delay is swept through.

        Returns:
            The frequency at which the target qubit will rotate when ZZ is zero
            based on the current experiment options.
        """
        delays = self.delays()
        freq = self.experiment_options.num_rotations / (max(delays) - min(delays))

        return freq

    def _template_circuits(
        self,
        frequency: Union[None, float, ParameterExpression] = None,
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
        delay = Parameter("delay")

        timing = BackendTiming(self.backend)

        frequency = frequency if frequency is not None else self.frequency()
        # frequency is always in units of Hz.  delay_freq has inverse units to
        # the units of `delay`.
        #
        # If the backend does not have a `dt`, the delays will be treated as in
        # units of seconds. Otherwise they will be in units of samples. For the
        # samples case, we multiply by `dt` so that `delay_freq` is in inverse
        # samples per cycle.
        if timing.delay_unit != "s":
            delay_freq = timing.dt * frequency
        else:
            delay_freq = frequency

        # Template circuit for series 0
        # Control qubit starting in |0> state, flipping to |1> in middle
        circ0 = QuantumCircuit(2, 1)
        circ0.metadata["series"] = "0"

        circ0.sx(0)

        circ0.barrier()
        circ0.delay(delay, 0, timing.delay_unit)
        circ0.barrier()

        circ0.x(0)
        circ0.x(1)

        circ0.barrier()
        circ0.delay(delay, 0, timing.delay_unit)
        circ0.barrier()

        circ0.rz(2 * np.pi * delay_freq * (2 * delay), 0)
        circ0.sx(0)
        # Flip control back to 0, so control qubit is in 0 for all circuits
        # when qubit 1 is measured.
        circ0.x(1)

        circ0.barrier()
        circ0.measure(0, 0)

        # Template circuit for series 1
        # Control qubit starting in |1> state, flipping to |0> in middle
        circ1 = QuantumCircuit(2, 1)
        circ1.metadata["series"] = "1"

        circ1.x(1)
        circ1.sx(0)

        circ1.barrier()
        circ1.delay(delay, 0, timing.delay_unit)
        circ1.barrier()

        circ1.x(0)
        circ1.x(1)

        circ1.barrier()
        circ1.delay(delay, 0, timing.delay_unit)
        circ1.barrier()

        circ1.rz(2 * np.pi * delay_freq * (2 * delay), 0)
        circ1.sx(0)

        circ1.barrier()
        circ1.measure(0, 0)

        return circ0, circ1

    def parametrized_circuits(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
        r"""Create circuits with parameters for numerical quantities

        This method is primarily intended for generating template circuits that
        visualize well. It inserts :class:`qiskit.circuit.Parameter`\ s for
        :math:`π` and `dt` as well the target qubit rotation frequency `f`.

        Return:
            Parameterized circuits for the case of the control qubit being in 0
            and in 1.
        """
        f_param = Parameter("f")
        dt = Parameter("dt")
        pi = Parameter("pi")

        freq = dt * pi * f_param / np.pi

        timing = BackendTiming(self.backend)
        if timing.dt is not None:
            freq = freq / timing.dt

        circs = self._template_circuits(frequency=freq)

        return circs

    def circuits(self) -> List[QuantumCircuit]:
        """Create circuits

        Returns:
            A list of circuits with a variable delay.
        """
        timing = BackendTiming(self.backend)

        circ0, circ1 = self._template_circuits()
        circs = []

        for delay in self.delays():
            for circ in (circ0, circ1):
                assigned = circ.assign_parameters(
                    {circ.parameters[0]: timing.round_delay(time=delay / 2)}, inplace=False
                )
                assigned.metadata["xval"] = 2 * timing.delay_time(time=delay / 2)
                circs.append(assigned)

        return circs
