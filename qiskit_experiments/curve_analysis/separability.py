# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Separability diagnostics for fits with nearly overlapping spectral features."""

from __future__ import annotations

import dataclasses

TASK_EXPONENTS = {
    "amplitudes_fixed_frequencies": 1,
    "frequencies": 2,
    "amplitudes_free_frequencies": 3,
}
"""Error amplification exponents for the three estimation tasks.

When two spectral features (resonance peaks, oscillation tones) sit a
distance ``separation`` apart and each has scale ``linewidth``, the
standard error of a fit grows like ``(linewidth / separation) ** p``
once the features overlap, with an exponent ``p`` that depends on what
is estimated: amplitudes with the two frequencies held fixed (p = 1),
the frequencies themselves (p = 2), or amplitudes with the frequencies
also free (p = 3). The exponents were measured across avoided crossings
on numerical relativity waveforms and quantum hardware, see
https://github.com/maiconburn/recoverability-criticality
(DOI 10.5281/zenodo.22156019); the frequency case matches the Fisher
scaling of arXiv:2605.16199 and the free-frequency amplitude case the
classical Prony super-resolution scaling.
"""


@dataclasses.dataclass(frozen=True)
class SeparabilityReport:
    """Result of :func:`peak_separability`.

    Attributes:
        ratio: ``separation / linewidth``.
        regime: ``"resolved"`` when the ratio is at least 2,
            ``"marginal"`` when it is between 0.5 and 2, and
            ``"unresolved"`` below 0.5. The thresholds are rules of
            thumb, not sharp boundaries.
        amplification: Expected error amplification for each task in
            :data:`TASK_EXPONENTS`, equal to ``max(1, 1 / ratio) ** p``.
    """

    ratio: float
    regime: str
    amplification: dict[str, float]


def peak_separability(separation: float, linewidth: float) -> SeparabilityReport:
    """Assess how well two nearby spectral features can be separated.

    Use this before or after fitting a two-peak or two-tone model to
    judge which fit outputs deserve trust. The amplification factors are
    asymptotic guides, valid up to prefactors of order one, so compare
    them between configurations rather than reading them as exact.

    Args:
        separation: Distance between the two feature positions, in the
            same units as ``linewidth`` (for example Hz).
        linewidth: Characteristic width of a single feature, for
            example the full width at half maximum.

    Returns:
        A :class:`SeparabilityReport` with the ratio, a qualitative
        regime label, and per-task error amplification factors.

    Raises:
        ValueError: If ``separation`` or ``linewidth`` is not positive.
    """
    if separation <= 0 or linewidth <= 0:
        raise ValueError("separation and linewidth must be positive.")

    ratio = separation / linewidth
    if ratio >= 2:
        regime = "resolved"
    elif ratio >= 0.5:
        regime = "marginal"
    else:
        regime = "unresolved"

    base = max(1.0, 1.0 / ratio)
    amplification = {task: base**p for task, p in TASK_EXPONENTS.items()}

    return SeparabilityReport(ratio=ratio, regime=regime, amplification=amplification)
