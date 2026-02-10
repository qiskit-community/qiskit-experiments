# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Experiment constants."""

from __future__ import annotations

import enum


class ResultQuality(enum.Enum):
    """Possible values for analysis result quality."""

    BAD = "bad"
    GOOD = "good"
    UNKNOWN = "unknown"

    @staticmethod
    def from_str(quality: str) -> ResultQuality:
        """Convert quality to a ResultQuality, defaulting to UNKNOWN"""
        try:
            result = ResultQuality(str(quality).lower())
        except ValueError:
            result = ResultQuality.UNKNOWN
        return result

    @staticmethod
    def to_str(quality: ResultQuality) -> str:
        """Convert quality to string, defaulting to "unknown" """
        if isinstance(quality, ResultQuality):
            return quality.value
        return "unknown"
