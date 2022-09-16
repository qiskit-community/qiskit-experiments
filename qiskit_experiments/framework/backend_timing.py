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
"""Backend timing helper functions"""

from math import gcd
from typing import Union

from qiskit import QiskitError
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BackendData


def lcm(int1: int, int2: int) -> int:
    """Least common multiple

    ``math.lcm`` was added in Python 3.9. This function should be replaced with
    ``from math import lcm`` after dropping support for Python 3.8.

    .. note::

        ``math.lcm`` supports an arbitrary number of arguments, but this
        version supports exactly two.
    """
    return int1 * int2 // gcd(int1, int2)


class BackendTiming:
    """Helper for calculating pulse and delay times for an experiment

    The methods and properties provided by this class help with calculating
    delay and pulse timing that depends on the timing constraints of the
    backend.

    When designing qubit characterization experiments, it is often necessary to
    deal with precise timing of pulses and delays. The fact that physical
    backends (i.e. not simulators) only support sampling time at intervals of
    ``dt`` complicates this process as times must be rounded. Besides the
    sampling time, there can be additional constraints like a minimum pulse
    length or a pulse granularity, which specifies the allowed increments of a
    pulse length in samples (i.e. for a granularity of 16 pulse lengths of 64
    and 80 samples are valid but not any number in between).

    Here are some specific problems that can occur when dealing timing
    constraints for pulses and delays:

    - An invalid pulse length or pulse start time could result in an error from
      the backend.
    - An invalid delay length could be rounded by the backend, and this
      rounding could lead to error in analysis that assumes the unrounded
      value.
    - An invalid delay length that requires rounding could trigger a new
      scheduling pass of a circuit, which is a computationally expensive
      process. Scheduling the circuit with valid timing to start out can avoid
      this rescheduling.
    - While there are separate alignment requirements for drive
      (``pulse_alignment``) and for measurement (``acquire_alignment``)
      channels, the nature of pulse and circuit instruction alignment can
      couple the timing of different instructions, resulting in improperly
      aligned instructions.  For example, consider this circuit:

      .. code-block:: python

          from qiskit import QuantumCircuit
          qc = QuantumCircuit(1, 1)
          qc.x(0)
          qc.delay(delay, 0)
          qc.x(0)
          qc.delay(delay2, 0)
          qc.measure(0, 0)

      Because the circuit instructions are all pushed together sequentially in
      time without extra delays, whether or not the ``measure`` instruction
      occurs at a valid time depends on the details of the circuit. In
      particular, since the ``x`` gates should have durations that are
      multiples of ``acquire_alignment`` (because ``granularity`` usually is),
      the ``measure`` start will occur at a time consistent with
      ``acquire_alignment`` when ``delay + delay2`` is a multiple of
      ``acquire_alignment``. Note that in the case of IBM Quantum backends,
      when ``acquire_alignment`` is not satisfied, there is no error reported
      by Qiskit or by the backend. Instead the measurement pulse is misaligned
      relative to the start of the signal acquisition, resulting in an
      incorrect phase and often an incorrect state discrimination.

    To help avoid these problems, :class:`.BackendTiming` provides methods for
    calculating pulse and delay durations in samples and seconds, for a given
    input duration in samples or seconds. If these values are used for all
    durations in a circuit, the alignment constraints should always be
    satisfied.

    As an example use-case for :class:`.BackendTiming`, consider a T1 experiment
    where delay times are specified in seconds and a
    :method:`.BaseExperiment.circuits`` method as follows:

    .. code-block:: python

        def circuits(self):
            # Pass experiment to BackendTiming
            timing = BackendTiming(self)

            circuits = []
            for delay in self.experiment_options.delays:
                circ = QuantumCircuit(1, 1)
                circ.x(0)
                # Convert delay into appropriate units for backend and also set
                # those units with delay_unit
                circ.delay(timing.circuit_delay(delay), 0, timing.delay_unit)
                circ.measure(0, 0)

                # Use delay_time to get the actual value in seconds that was
                # set on the backend for the xval rather than the delay
                # variable's nominal value.
                circ.metadata = {
                    "unit": "s",
                    "xval": timing.delay_time(delay),
                }

                circuits.append(circ)

    As another example, consider a time Rabi experiment where the width of a
    pulse in a schedule is stretched:

    .. code-block:: python

        from qiskit import pulse
        from qiskit.circuit import Gate, Parameter

        def circuits(self):
            chan = pulse.DriveChannel(0)
            dur = Paramater("duration")

            with pulse.build() as sched:
                pulse.play(pulse.Gaussina(duration=dur, amp=1, sigma=dur / 4), chan)

            gate = Gate("Rabi", num_qubits=1, params=[dur])

            template_circ = QuantumCircuit(1, 1)
            template_circ.append(gate, [0])
            template_circ.measure(0, 0)
            template_circ.add_calibration(gate, (0,), sched)

            # Pass experiment to BackendTiming
            timing = BackendTiming(self)

            circs = []
            for duration in self.experiment_options.durations:
                # Calculate valid sample number closest to this duration
                circ = template_circ.assign_parameters(
                    {dur: timing.pulse_samples(duration)},
                    inplace=False,
                )
                # Track corresponding duration in seconds for the pulse
                circ.metadata = {
                    "xval": timing.pulse_time(duration),
                    "unit": "s",
                }



    .. note::

        For delay duration, the least common multiple of ``pulse_alignment``
        and ``acquire_alignment`` is used as the granularity. Thus, in the
        ``acquire_alignment`` example above, ``delay`` and ``delay2`` are each
        a multiple of ``acquire_alignment`` and so the sum always is. This
        approach excludes some valid circuits (like each delay being half of
        ``acquire_alignment``) but has the benefit of always being valid
        without detailed analysis of the full circuit.
    """

    def __init__(self, backend: Backend):
        """Initialize backend timing object

        Args:
            experiment: the experiment to provide timing help for
        """
        backend_data = BackendData(backend)

        # Pull all the timing data from the backend
        self._acquire_alignment = backend_data.acquire_alignment
        self._granularity = backend_data.granularity
        self._min_length = backend_data.min_length
        self._pulse_alignment = backend_data.pulse_alignment
        #: The backend's ``dt`` value, copied to :class:`.BackendTiming` for convenience
        self.dt = backend_data.dt

    @property
    def delay_unit(self) -> str:
        """The delay unit for the current backend

        "dt" is used if dt is present in the backend configuration. Otherwise
        "s" is used.
        """
        if self.dt is not None:
            return "dt"

        return "s"

    def circuit_delay(self, time: float) -> Union[int, float]:
        """Delay duration close to ``time`` and consistent with timing constraints

        This method produces the value to pass for the ``duration`` of a
        ``Delay`` instruction of a ``QuantumCircuit`` schedule so that the
        delay fills the time until the next valid pulse, assuming the ``Delay``
        instruction begins on a sample that is also valid for a pulse to begin
        on.

        The pulse timing constraints of the backend are considered in order to
        give a number of samples closest to ``time`` plus however many more
        samples are needed to get to the next valid sample for the start of a
        pulse in a subsequent instruction. The least common multiple of the
        pulse and acquire alignment values is used in order to ensure that
        either type of pulse will be aligned.

        If :meth:`.BackendTiming.delay_unit` is ``s``, ``time`` is
        returned directly. Typically, this is the case for a simulator where
        converting to sample number is not needed.

        Args:
            time: The nominal delay time to convert in seconds

        Returns:
            The delay duration in samples if :meth:`.BackendTiming.delay_unit`
            is ``dt``. Other return ``time``.
        """
        if self.delay_unit == "s":
            return time

        return self.schedule_delay(time)

    def schedule_delay(self, time: float) -> int:
        """Valid delay value in samples to use in a pulse schedule for  ``time``

        The pulse timing constraints of the backend are considered in order to
        give a number of samples closest to ``time`` plus however many more
        samples are needed to get to the next valid sample for the start of a
        pulse in a subsequent instruction. The least common multiple of the
        pulse and acquire alignment values is used in order to ensure that
        either type of pulse will be aligned.

        Args:
            time: The nominal delay time to convert in seconds

        Returns:
            The delay duration in samples to pass to a ``Delay`` instruction in
            Qiskit pulse schedule

        Raises:
            QiskitError: The backend does not include a dt value.
        """
        if self.dt is None:
            raise QiskitError("Backend has no dt value.")

        granularity = lcm(self._pulse_alignment, self._acquire_alignment)

        samples = int(round(time / self.dt / granularity) * granularity)

        return samples

    def pulse_samples(self, time: float) -> int:
        """The number of samples giving a valid pulse duration closest to ``time``

        The multiple of the pulse granularity giving the time closest to but
        higher than ``time`` is used. The returned value is always at least the
        backend's ``min_length``.

        Args:
            time: Pulse duration in seconds

        Returns:
            The number of samples corresponding to ``time``

        Raises:
            QiskitError: The backend does not include a dt value.
        """
        if self.dt is None:
            raise QiskitError("Backend has no dt value.")

        return self.round_pulse_samples(time / self.dt)

    def round_pulse_samples(self, samples: Union[float, int]) -> int:
        """Round a nominal pulse sample duration to a valid number

        The multiple of the pulse granularity giving the samples closest to but
        higher than ``samples`` is used. The returned value is always at least
        the backend's ``min_length``.

        Args:
            samples: Nominal pulse duration

        Returns:
            A sample duration close to ``samples`` and consistent with the
            backend's timing constraints

        Raises:
            QiskitError: If the algorithm used to calculate the pulse length
                produces a length that is not commensurate with the pulse or
                acquire alignment values. This should not happen unless the
                alignment constraints provided by the backend do not fit the
                assumptions that the algorithm makes.
        """
        samples = int(round(samples / self._granularity)) * self._granularity
        samples = max(samples, self._min_length)

        pulse_alignment = self._pulse_alignment
        acquire_alignment = self._acquire_alignment

        if samples % pulse_alignment != 0:
            raise QiskitError(
                "Pulse duration calculation does not match pulse alignment constraints!"
            )

        if samples % acquire_alignment != 0:
            raise QiskitError(
                "Pulse duration calculation does not match acquire alignment constraints!"
            )

        return samples

    def delay_time(self, time: float) -> float:
        """The closest actual delay time in seconds greater than ``time``

        If the backend reports ``dt``, this method uses
        :meth:`.BackendTiming.schedule_delay` and converts the resultback into
        seconds. Otherwise, it just returns ``time`` directly.

        Args:
            time: The nominal delay time to be rounded

        Returns:
            The realizable delay time in seconds
        """
        if self.delay_unit == "s":
            return time

        return self.dt * self.schedule_delay(time)

    def pulse_time(self, time: float) -> float:
        """The closest valid pulse duration greater than ``time`` in seconds

        This method uses :meth:`.BackendTiming.pulse_samples` and then
        converts back into seconds.

        Args:
            time: The nominal pulse time to be rounded

        Returns:
            The realizable pulse time in seconds

        Raises:
            QiskitError: The backend does not include a dt value.
        """
        if self.dt is None:
            raise QiskitError("Backend has no dt value.")

        return self.dt * self.pulse_samples(time)
