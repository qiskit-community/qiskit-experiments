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

import warnings
from typing import Optional, Union

from qiskit import QiskitError
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BackendData


class BackendTiming:
    """Helper for calculating pulse and delay times for an experiment

    The methods and properties provided by this class help with calculating
    delay timing that depends on the timing constraints of the backend.

    When designing qubit characterization experiments, it is often necessary to
    deal with precise timing of delays. The fact that physical backends (i.e.
    not simulators) only support sampling time at intervals of ``dt``
    complicates this process as times must be rounded. Besides the
    sampling time, there can be additional constraints like a
    granularity, which specifies the allowed increments of a
    delay length in samples (i.e., for a granularity of 16, delay lengths of 64
    and 80 samples are valid but not any number in between).

    Here are some specific problems that can occur when dealing with timing
    constraints for pulses and delays:

    - An invalid delay length could be rounded by the backend, and this
      rounding could lead to error in analysis that assumes the unrounded
      value.
    - An invalid delay length that requires rounding could trigger a new
      scheduling pass of a circuit during transpilation, which is a
      computationally expensive process. Scheduling the circuit with valid
      timing to start out can avoid this rescheduling.

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
            acquire_alignment: Optional. Deprecated and unused.
            granularity: Optional. Constraint for the pulse samples granularity
                in units of dt. Defaults to the backend value.
            min_length: Optional. Deprecated and unused.
            pulse_alignment: Optional. Deprecated and unused.
            dt: Optional. Time interval of pulse samples. Default to the backend value.
        """
        backend_data = BackendData(backend)

        # Pull all the timing data from the backend
        self._granularity = granularity or backend_data.granularity
        #: The backend's ``dt`` value, copied to :class:`.BackendTiming` for convenience
        self.dt = dt or backend_data.dt

        if min_length is not None or acquire_alignment is not None or pulse_alignment is not None:
            warnings.warn(
                "Arguments acquire_alignment, min_length, and pulse_alignment "
                "are no longer used by BackendTiming."
            )

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

        samples_out = int(round(samples / self._granularity) * self._granularity)

        return samples_out

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
