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
from typing import Optional, Union

from qiskit import QiskitError
from qiskit.providers.backend import Backend
from qiskit.utils.deprecation import deprecate_func

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
    pulse length in samples (i.e., for a granularity of 16, pulse lengths of 64
    and 80 samples are valid but not any number in between).

    Here are some specific problems that can occur when dealing with timing
    constraints for pulses and delays:

    - An invalid pulse length or pulse start time could result in an error from
      the backend.
    - An invalid delay length could be rounded by the backend, and this
      rounding could lead to error in analysis that assumes the unrounded
      value.
    - An invalid delay length that requires rounding could trigger a new
      scheduling pass of a circuit during transpilation, which is a
      computationally expensive process. Scheduling the circuit with valid
      timing to start out can avoid this rescheduling.
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
      particular, since the ``x`` gates typically have durations that are
      multiples of ``acquire_alignment`` (because ``granularity`` usually is),
      the ``measure`` start will occur at a time consistent with
      ``acquire_alignment`` when ``delay + delay2`` is a multiple of
      ``acquire_alignment``. Note that in the case of IBM Quantum backends,
      when ``acquire_alignment`` is not satisfied, there is no error reported
      by Qiskit or by the backend. Instead the measurement pulse is misaligned
      relative to the start of the signal acquisition, resulting in an
      incorrect phase and often an incorrect state discrimination.

    To help avoid these problems, :class:`.BackendTiming` provides methods for
    calculating pulse and delay durations. These methods work with samples and
    seconds as appropriate. If these methods are used for all durations in a
    circuit, the alignment constraints should always be satisfied.

    .. note::

        For delay duration, the least common multiple of ``pulse_alignment``
        and ``acquire_alignment`` is used as the granularity. Thus, in the
        example above about the coupling between ``pulse_alignment`` and
        ``acquire_alignment`` , ``delay`` and ``delay2`` would each be rounded
        to a multiple of ``acquire_alignment`` and so the sum would always be a
        multiple of each alignment value as well. This approach modifies  some
        valid circuits (like each delay being half of ``acquire_alignment``)
        but has the benefit of always being valid without detailed analysis of
        the full circuit.

    As an example use-case for :class:`.BackendTiming`, consider a T1 experiment
    where delay times are specified in seconds in a
    :meth:`qiskit_experiments.framework.BaseExperiment.circuits` method as
    follows:

    .. code-block:: python

        def circuits(self):
            # Pass backend to BackendTiming
            timing = BackendTiming(self.backend)

            circuits = []
            # delays is a list of delay values in seconds
            for delay in self.experiment_options.delays:
                circ = QuantumCircuit(1, 1)
                circ.x(0)
                # Convert delay into appropriate units for backend and also set
                # those units with delay_unit
                circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
                circ.measure(0, 0)

                # Use delay_time to get the actual value in seconds that was
                # set on the backend for the xval rather than the delay
                # variable's nominal value.
                circ.metadata = {
                    "unit": "s",
                    "xval": timing.delay_time(time=delay),
                }

                circuits.append(circ)

    As another example, consider a time Rabi experiment where the width of a
    pulse in a schedule is stretched:

    .. code-block:: python

        from qiskit import pulse
        from qiskit.circuit import Gate, Parameter


        def circuits(self):
            chan = pulse.DriveChannel(0)
            dur = Parameter("duration")

            with pulse.build() as sched:
                pulse.play(pulse.Gaussian(duration=dur, amp=1, sigma=dur / 4), chan)

            gate = Gate("Rabi", num_qubits=1, params=[dur])

            template_circ = QuantumCircuit(1, 1)
            template_circ.append(gate, [0])
            template_circ.measure(0, 0)
            template_circ.add_calibration(gate, (0,), sched)

            # Pass backend to BackendTiming
            timing = BackendTiming(self.backend)

            circs = []
            # durations is a list of pulse durations in seconds
            for duration in self.experiment_options.durations:
                # Calculate valid sample number closest to this duration
                circ = template_circ.assign_parameters(
                    {dur: timing.round_pulse(time=duration)},
                    inplace=False,
                )
                # Track corresponding duration for the pulse in seconds
                circ.metadata = {
                    "xval": timing.pulse_time(time=duration),
                    "unit": "s",
                }
    """

    def __init__(
        self,
        backend: Backend,
        *,
        acquire_alignment: Optional[int] = None,
        granularity: Optional[int] = None,
        min_length: Optional[int] = None,
        pulse_alignment: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        """Initialize backend timing object

        .. note::
            Backend may not accept user defined constraint value.
            One may want to provide these values when the constraints data is missing in the backend,
            or in some situation you can intentionally ignore the constraints.
            Invalid constraint values may break experiment circuits, resulting in the
            failure in or unexpected results from the execution.

        Args:
            backend: the backend to provide timing help for.
            acquire_alignment: Optional. Constraint for the acquisition instruction alignment
                in units of dt. Default to the backend value.
            granularity: Optional. Constraint for the pulse samples granularity
                in units of dt. Defaults to the backend value.
            min_length: Optional. Constraint for the minimum pulse samples
                in units of dt. Defaults to the backend value.
            pulse_alignment: Optional. Constraint for the pulse play instruction alignment
                in units of dt. Default to the backend value.
            dt: Optional. Time interval of pulse samples. Default to the backend value.
        """
        backend_data = BackendData(backend)

        # Pull all the timing data from the backend
        self._acquire_alignment = acquire_alignment or backend_data.acquire_alignment
        self._granularity = granularity or backend_data.granularity
        self._min_length = min_length or backend_data.min_length
        self._pulse_alignment = pulse_alignment or backend_data.pulse_alignment
        #: The backend's ``dt`` value, copied to :class:`.BackendTiming` for convenience
        self.dt = dt or backend_data.dt

    @property
    def delay_unit(self) -> str:
        """The delay unit for the current backend

        "dt" is used if dt is present in the backend configuration. Otherwise
        "s" is used.
        """
        if self.dt is not None:
            return "dt"

        return "s"

    def round_delay(
        self, *, time: Optional[float] = None, samples: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """Delay duration closest to input and consistent with timing constraints

        This method produces the value to pass for the ``duration`` of a
        ``Delay`` instruction of a ``QuantumCircuit`` so that the delay fills
        the time until the next valid pulse, assuming the ``Delay`` instruction
        begins on a sample that is also valid for a pulse to begin on.

        The pulse timing constraints of the backend are considered in order to
        give the number of samples closest to the input (either ``time`` or
        ``samples``) for the start of a pulse in a subsequent instruction to be
        valid. The delay value in samples is rounded to the least common
        multiple of the pulse and acquire alignment values in order to ensure
        that either type of pulse will be aligned.

        If :meth:`.BackendTiming.delay_unit` is ``s``, ``time`` is
        returned directly. Typically, this is the case for a simulator where
        converting to sample number is not needed.

        Args:
            time: The nominal delay time to convert in seconds
            samples: The nominal delay time to convert in samples

        Returns:
            The delay duration in samples if :meth:`.BackendTiming.delay_unit`
            is ``dt``. Otherwise return ``time``.

        Raises:
            QiskitError: If either both ``time`` and ``samples`` are passed or
                neither is passed.
        """
        if time is None and samples is None:
            raise QiskitError("Either time or samples must be a numerical value.")
        if time is not None and samples is not None:
            raise QiskitError("Only one of time and samples can be a numerical value.")

        if self.dt is None and time is not None:
            return time

        if samples is None:
            samples = time / self.dt

        granularity = lcm(self._pulse_alignment, self._acquire_alignment)

        samples_out = int(round(samples / granularity) * granularity)

        return samples_out

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, utility functions involving "
            "pulse like this one have been deprecated."
        ),
    )
    def round_pulse(
        self, *, time: Optional[float] = None, samples: Optional[Union[int, float]] = None
    ) -> int:
        """The number of samples giving the valid pulse duration closest to the input

        The multiple of the pulse granularity giving the time closest to the
        input (either ``time`` or ``samples``) is used. The returned value is
        always at least the backend's ``min_length``.

        Args:
            time: Nominal pulse duration in seconds
            samples: Nominal pulse duration in samples

        Returns:
            The number of samples corresponding to the input

        Raises:
            QiskitError: If either both ``time`` and ``samples`` are passed or
                neither is passed.
            QiskitError: The backend does not include a dt value.
            QiskitError: If the algorithm used to calculate the pulse length
                produces a length that is not commensurate with the pulse or
                acquire alignment values. This should not happen unless the
                alignment constraints provided by the backend do not fit the
                assumptions that the algorithm makes.
        """
        if time is None and samples is None:
            raise QiskitError("Either time or samples must be a numerical value.")
        if time is not None and samples is not None:
            raise QiskitError("Only one of time and samples can be a numerical value.")

        if self.dt is None:
            raise QiskitError("Backend has no dt value.")

        if samples is None:
            samples = time / self.dt

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

    def delay_time(
        self, *, time: Optional[float] = None, samples: Optional[Union[int, float]] = None
    ) -> float:
        """The closest valid delay time in seconds to the input

        If the backend reports ``dt``, this method uses
        :meth:`.BackendTiming.round_delay` and converts the result back into
        seconds. Otherwise, if ``time`` was passed, it is returned directly.

        Args:
            time: The nominal delay time to convert in seconds
            samples: The nominal delay time to convert in samples

        Returns:
            The realizable delay time in seconds

        Raises:
            QiskitError: If either both ``time`` and ``samples`` are passed or
                neither is passed.
        """
        if self.delay_unit == "s":
            return time

        return self.dt * self.round_delay(time=time, samples=samples)

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, utility functions involving "
            "pulse like this one have been deprecated."
        ),
    )
    def pulse_time(
        self, *, time: Optional[float] = None, samples: Optional[Union[int, float]] = None
    ) -> float:
        """The closest valid pulse duration to the input in seconds

        This method uses :meth:`.BackendTiming.round_pulse` and then
        converts back into seconds.

        Args:
            time: Nominal pulse duration in seconds
            samples: Nominal pulse duration in samples

        Returns:
            The realizable pulse time in seconds

        Raises:
            QiskitError: If either both ``time`` and ``samples`` are passed or
                neither is passed.
            QiskitError: The backend does not include a dt value.
            QiskitError: If the algorithm used to calculate the pulse length
                produces a length that is not commensurate with the pulse or
                acquire alignment values. This should not happen unless the
                alignment constraints provided by the backend do not fit the
                assumptions that the algorithm makes.
        """
        return self.dt * self.round_pulse(time=time, samples=samples)
