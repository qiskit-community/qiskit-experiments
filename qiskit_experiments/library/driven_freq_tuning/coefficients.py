# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Coefficients characterizing Stark shift."""

from __future__ import annotations
from typing import Any

import numpy as np

from qiskit.providers.backend import Backend
from qiskit_ibm_experiment.service import IBMExperimentService
from qiskit_ibm_experiment.exceptions import IBMApiError

from qiskit_experiments.framework.json import ExperimentDecoder
from qiskit_experiments.framework.backend_data import BackendData
from qiskit_experiments.framework.experiment_data import ExperimentData


class StarkCoefficients:
    """A collection of coefficients characterizing Stark shift."""

    def __init__(
        self,
        pos_coef_o1: float,
        pos_coef_o2: float,
        pos_coef_o3: float,
        neg_coef_o1: float,
        neg_coef_o2: float,
        neg_coef_o3: float,
        offset: float,
    ):
        """Create new coefficients object.

        Args:
            pos_coef_o1: The first order shift coefficient on positive amplitude.
            pos_coef_o2: The second order shift coefficient on positive amplitude.
            pos_coef_o3: The third order shift coefficient on positive amplitude.
            neg_coef_o1: The first order shift coefficient on negative amplitude.
            neg_coef_o2: The second order shift coefficient on negative amplitude.
            neg_coef_o3: The third order shift coefficient on negative amplitude.
            offset: Offset frequency.
        """
        self.pos_coef_o1 = pos_coef_o1
        self.pos_coef_o2 = pos_coef_o2
        self.pos_coef_o3 = pos_coef_o3
        self.neg_coef_o1 = neg_coef_o1
        self.neg_coef_o2 = neg_coef_o2
        self.neg_coef_o3 = neg_coef_o3
        self.offset = offset

    def positive_coeffs(self) -> list[float]:
        """Positive coefficients."""
        return [self.pos_coef_o3, self.pos_coef_o2, self.pos_coef_o1]

    def negative_coeffs(self) -> list[float]:
        """Negative coefficients."""
        return [self.neg_coef_o3, self.neg_coef_o2, self.neg_coef_o1]

    def convert_freq_to_amp(
        self,
        freqs: np.ndarray,
    ) -> np.ndarray:
        """A helper function to convert Stark frequency to amplitude.

        Args:
            freqs: Target frequency shifts to compute required Stark amplitude.

        Returns:
            Estimated Stark amplitudes to induce input frequency shifts.

        Raises:
            ValueError: When amplitude value cannot be solved.
        """
        amplitudes = np.zeros_like(freqs)
        for idx, freq in enumerate(freqs):
            shift = freq - self.offset
            if np.isclose(shift, 0.0):
                amplitudes[idx] = 0.0
                continue
            if shift > 0:
                fit = [*self.positive_coeffs(), -shift]
            else:
                fit = [*self.negative_coeffs(), -shift]
            amp_candidates = np.roots(fit)
            # Because the fit function is third order, we get three solutions here.
            criteria = np.all(
                [
                    # Frequency shift and tone have the same sign by definition
                    np.sign(amp_candidates.real) == np.sign(shift),
                    # Tone amplitude is a real value
                    np.isclose(amp_candidates.imag, 0.0),
                    # The absolute value of tone amplitude must be less than 1.0 + 10 mp
                    np.abs(amp_candidates.real) < 1.0 + 10 * np.finfo(float).eps,
                ],
                axis=0,
            )
            valid_amps = amp_candidates[criteria]
            if len(valid_amps) == 0:
                raise ValueError(f"Stark shift at frequency value of {freq} Hz is not available.")
            if len(valid_amps) > 1:
                # We assume a monotonic trend but sometimes a large third-order term causes
                # inflection point and inverts the trend in larger amplitudes.
                # In this case we would have more than one solution, but we can
                # take the smallest amplitude before reaching to the inflection point.
                before_inflection = np.argmin(np.abs(valid_amps.real))
                valid_amp = float(valid_amps[before_inflection].real)
            else:
                valid_amp = float(valid_amps[0].real)
            amplitudes[idx] = min(valid_amp, 1.0)
        return amplitudes

    def convert_amp_to_freq(
        self,
        amps: np.ndarray,
    ) -> np.ndarray:
        """A helper function to convert Stark amplitude to frequency shift.

        Args:
            amps: Amplitude values to convert into frequency shift.

        Returns:
              Calculated frequency shift at given Stark amplitude.
        """
        pos_fit = np.poly1d([*self.positive_coeffs(), self.offset])
        neg_fit = np.poly1d([*self.negative_coeffs(), self.offset])

        return np.where(amps > 0, pos_fit(amps), neg_fit(amps))

    def find_min_max_frequency(
        self,
        min_amp: float,
        max_amp: float,
    ) -> tuple[float, float]:
        """A helper function to estimate maximum frequency shift within given amplitude budget.

        Args:
            min_amp: Minimum Stark amplitude.
            max_amp: Maximum Stark amplitude.

        Returns:
            Minimum and maximum frequency shift available within the amplitude range.
        """
        trim_amps = []
        for amp in [min_amp, max_amp]:
            if amp > 0:
                fit = self.positive_coeffs()
            else:
                fit = self.negative_coeffs()
            # Solve for inflection points by computing the point where derivative becomes zero.
            solutions = np.roots([deriv * coeff for deriv, coeff in zip((3, 2, 1), fit)])
            inflection_points = solutions[
                (solutions.imag == 0) & (np.sign(solutions) == np.sign(amp))
            ]
            if len(inflection_points) > 0:
                # When multiple inflection points are found, use the most outer one.
                # There could be a small inflection point around amp=0,
                # when the first order term is significant.
                amp = min([amp, max(inflection_points, key=abs)], key=abs)
            trim_amps.append(amp)
        return tuple(self.convert_amp_to_freq(np.asarray(trim_amps)))

    def __str__(self):
        # Short representation for dataframe
        return "StarkCoefficients(...)"

    def __eq__(self, other):
        return all(
            [
                self.pos_coef_o1 == other.pos_coef_o1,
                self.pos_coef_o2 == other.pos_coef_o2,
                self.pos_coef_o3 == other.pos_coef_o3,
                self.neg_coef_o1 == other.neg_coef_o1,
                self.neg_coef_o2 == other.neg_coef_o2,
                self.neg_coef_o3 == other.neg_coef_o3,
                self.offset == other.offset,
            ]
        )

    def __json_encode__(self) -> dict[str, Any]:
        return {
            "class": "StarkCoefficients",
            "data": {
                "pos_coef_o1": self.pos_coef_o1,
                "pos_coef_o2": self.pos_coef_o2,
                "pos_coef_o3": self.pos_coef_o3,
                "neg_coef_o1": self.neg_coef_o1,
                "neg_coef_o2": self.neg_coef_o2,
                "neg_coef_o3": self.neg_coef_o3,
                "offset": self.offset,
            },
        }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "StarkCoefficients":
        if not value.get("class", None) == "StarkCoefficients":
            raise ValueError("JSON decoded value for StarkCoefficients is not valid class type.")
        return StarkCoefficients(**value.get("data", {}))


def retrieve_coefficients_from_service(
    service: IBMExperimentService,
    backend_name: str,
    qubit: int,
) -> StarkCoefficients:
    """Retrieve StarkCoefficients object from experiment service.

    Args:
        service: IBM Experiment service instance interfacing with result database.
        backend_name: Name of target backend.
        qubit: Index of qubit.

    Returns:
        StarkCoefficients object.

    Raises:
        RuntimeError: When stark_coefficients entry doesn't exist in the service.
    """
    try:
        retrieved = service.analysis_results(
            device_components=[f"Q{qubit}"],
            result_type="stark_coefficients",
            backend_name=backend_name,
            sort_by=["creation_datetime:desc"],
            json_decoder=ExperimentDecoder,
            # Returns the latest value only. IBM service returns 10 entries by default.
            # This could contain old data from previous version, which might not be deserialized.
            limit=1,
        )
    except (IBMApiError, ValueError) as ex:
        raise RuntimeError(
            f"Failed to retrieve the result of stark_coefficients: {ex.message}"
        ) from ex
    if len(retrieved) == 0:
        raise RuntimeError(
            "Analysis result of stark_coefficients is not found in the "
            "experiment service. Run and save the result of StarkRamseyXYAmpScan."
        )

    result_data_dict = retrieved[0].result_data
    if "_value" in result_data_dict:
        # In IBM Experiment service, the result_data["value"] returns
        # a display value for the experiment service webpage.
        # Original data is stored in "_value".
        # TODO: this must be handled by experiment service.
        return result_data_dict["_value"]
    return result_data_dict["value"]


def retrieve_coefficients_from_backend(
    backend: Backend,
    qubit: int,
) -> StarkCoefficients:
    """Retrieve StarkCoefficients object from the Qiskit backend.

    Args:
        backend: Qiskit backend object.
        qubit: Index of qubit.

    Returns:
        StarkCoefficients object.

    Raises:
        RuntimeError: When experiment service cannot be loaded from backend.
    """
    name = BackendData(backend).name
    service = ExperimentData.get_service_from_backend(backend)

    if service is None:
        raise RuntimeError(f"Valid experiment service is not found for the backend {name}.")

    return retrieve_coefficients_from_service(service, name, qubit)
