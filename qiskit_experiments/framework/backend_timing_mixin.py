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

import math
from typing import Union

from qiskit import QiskitError


class BackendTimingMixin:
    """Mixin class  to provide timing helper methods

    The methods and properties provided by this class help with calculating
    delay and pulse timing that depends on the timing constraints of the
    backend. They abstract away the necessary accessing of the backend object.
    This class is intended as a mixin for
    :meth:`~qiskit_experiments.framework.BaseExperiment`.

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

    def _delay_duration(self, time: float) -> Union[int, float]:
        """Delay duration close to ``time`` and consistent with timing constraints

        This method produces the value to pass for the ``duration`` of a
        ``Delay`` instruction of a ``QuantumCircuit`` so that the delay fills
        the time until the next valid pulse, assuming the ``Delay`` instruction
        begins on a sample that is also valid for pulse to begin on.

        The pulse timing constraints of the backend are considered in order to
        give a number of samples closest to ``time`` plus however many more
        samples are needed to get to the next valid sample for the start of a
        pulse in a subsequent instruction. The least common multiple of the
        pulse and acquire alignment values is used in order to ensure that
        either type of pulse will be aligned.

        If :meth:`.BackendTimingMixin._delay_unit` is ``s``, ``time`` is
        returned directly. Typically, this is the case for a simulator where
        converting to sample number is not needed.

        Args:
            time: The nominal delay time to convert

        Returns:
            The delay duration in samples if :meth:`.BackendTimingMixin._delay_unit`
            is ``dt``. Other return ``time``.
        """
        if self._delay_unit == "s":
            return time

        timing_constraints = getattr(self.backend.configuration(), "timing_constraints", {})
        pulse_alignment = timing_constraints.get("pulse_alignment", 1)
        acquire_alignment = timing_constraints.get("acquire_alignment", 1)

        # Replace with math.lcm(pulse_alignment, acquire_alignment) when
        # dropping support for Python 3.8
        granularity = (
            pulse_alignment * acquire_alignment // math.gcd(pulse_alignment, acquire_alignment)
        )

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

        return samples

    def _delay_time(self, time: float) -> float:
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

        return self._dt * self._delay_duration(time)

    def _pulse_time(self, time: float) -> float:
        """The closest hardware-realizable pulse duration to ``time`` in seconds

        This method uses :meth:`.BackendTimingMixin._pulse_duration` and then
        converts back into seconds.

        Args:
            time: The nominal pulse time to be rounded

        Returns:
            The realizable pulse time in seconds
        """
        return self._dt * self._pulse_duration(time)
