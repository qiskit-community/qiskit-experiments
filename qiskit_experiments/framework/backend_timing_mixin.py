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

from qiskit_experiments.framework import BaseExperiment
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

    .. note::

        The methods in this class assume that the ``backend`` attribute is
        constant. Methods should not call methods of this class before and
        after modifying the ``backend`` attribute and expect consistent
        results. In particular, when the backend is not set, no ``dt`` value is
        available and :meth:`.BackendTiming.circuit_delay` will return the
        delay time in seconds whereas once the backend is set the value
        returned will be in units of ``dt`` rounded according to the alignment
        constraints.
    """

    def __init__(self, experiment: BaseExperiment):
        """Initialize backend timing object

        Args:
            experiment: the experiment to provide timing help for
        """
        self.experiment = experiment

        self._backend: Union[Backend, None] = None
        self._backend_data_cached: Union[BackendData, None] = None

    @property
    def _backend_data(self) -> BackendData:
        """Backend data associated with experiment

        Returns:
            The BackendData object associated with the experiment

        Raises:
            QiskitError: if the backend is not set on the experiment
        """
        if self.experiment.backend is None:
            raise QiskitError("Backend not set on experiment!")

        if self.experiment.backend != self._backend:
            self._backend = self.experiment.backend
            self._backend_data_cached = BackendData(self._backend)

        return self._backend_data_cached

    @property
    def delay_unit(self) -> str:
        """The delay unit for the current backend

        "dt" is used if dt is present in the backend configuration. Otherwise
        "s" is used.
        """
        if self._backend_data.dt is not None:
            return "dt"

        return "s"

    @property
    def _dt(self) -> float:
        """Backend dt value

        This property wraps ``_backend_data.dt`` in order to give a more
        specific error message when trying to use ``dt`` with a backend that
        does not provide it rather than just giving a ``TypeError`` about
        ``NoneType``. As this raises an exception when ``dt`` is not set, it
        likely should not be used by external code using
        :class:`.BackendTiming`.

        Raises:
            QiskitError: The backend does not include a dt value.
        """
        if self._backend_data.dt is not None:
            return float(self._backend_data.dt)

        raise QiskitError("Backend has no dt value.")

    def circuit_delay(self, time: float) -> Union[int, float]:
        """Delay duration close to ``time`` and consistent with timing constraints

        This method produces the value to pass for the ``duration`` of a
        ``Delay`` instruction of a ``QuantumCircuit`` or a pulse schedule so
        that the delay fills the time until the next valid pulse, assuming the
        ``Delay`` instruction begins on a sample that is also valid for a pulse
        to begin on.

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

    def schedule_delay(self, time: float) -> int
        pulse_alignment = self._backend_data.pulse_alignment
        acquire_alignment = self._backend_data.acquire_alignment

        granularity = lcm(pulse_alignment, acquire_alignment)

        samples = int(round(time / self._dt / granularity) * granularity)

        return samples

    def pulse_samples(self, time: float) -> int:
        """The number of samples giving a valid pulse duration closest to ``time``

        Args:
            time: Pulse duration in seconds

        Returns:
            The number of samples corresponding to ``time``

        Raises:
            QiskitError: If the algorithm used to calculate the pulse length
                produces a length that is not commensurate with the pulse or
                acquire alignment values. This should not happen unless the
                alignment constraints provided by the backend do not fit the
                assumptions that the algorithm makes.
        """
        granularity = self._backend_data.granularity
        min_length = self._backend_data.min_length

        samples = int(round(time / self._dt / granularity)) * granularity
        samples = max(samples, min_length)

        pulse_alignment = self._backend_data.pulse_alignment
        acquire_alignment = self._backend_data.acquire_alignment

        if samples % pulse_alignment != 0:
            raise QiskitError("Pulse duration calculation does not match pulse alignment constraints!")

        if samples % acquire_alignment != 0:
            raise QiskitError("Pulse duration calculation does not match acquire alignment constraints!")

        return samples

    def delay_time(self, time: float) -> float:
        """The closest actual delay time in seconds greater than ``time``

        This method uses :meth:`.BackendTiming.instruction_delay` and then
        converts back into seconds.

        Args:
            time: The nominal delay time to be rounded

        Returns:
            The realizable delay time in seconds
        """
        if self.delay_unit == "s":
            return time

        return self._dt * self.schedule_delay(time)

    def pulse_time(self, time: float) -> float:
        """The closest hardware-realizable pulse duration greater than ``time`` in seconds

        This method uses :meth:`.BackendTiming.pulse_samples` and then
        converts back into seconds.

        Args:
            time: The nominal pulse time to be rounded

        Returns:
            The realizable pulse time in seconds
        """
        return self._dt * self.pulse_samples(time)
