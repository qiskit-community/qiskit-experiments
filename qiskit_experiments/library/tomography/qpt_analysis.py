# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum process tomography analysis
"""
from qiskit_experiments.framework import Options
from .basis import PauliMeasurementBasis, PauliPreparationBasis
from .tomography_analysis import TomographyAnalysis


class ProcessTomographyAnalysis(TomographyAnalysis):
    """Quantum state and process tomography experiment analysis.

    # section: overview

        Fitter Functions

        Built-in fitter functions may be selected using the following string
        labels, refer to the corresponding functions documentation for additional
        details on the fitters.

        * ``"linear_inversion"``:
          :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion` (Default)
        * ``"scipy_linear_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.scipy_linear_lstsq`
        * ``"cvxpy_linear_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_linear_lstsq`
        * ``"scipy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.scipy_gaussian_lstsq`
        * ``"cvxpy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_gaussian_lstsq`

        PSD Rescaling

        For fitters that do not constrain the reconstructed state to be
        `positive-semidefinite` (PSD) we construct the maximum-likelihood
        nearest PSD state under the assumption of Gaussian measurement noise
        using the rescaling method in Reference [1]. For fitters that already
        support PSD constraints this option can be disabled by setting
        ``rescale_positive=False``.

    # section: note
        Fitters starting with ``"cvxpy_*"`` require the optional CVXPY Python
        package to be installed.

    # section: warning
        The API for tomography fitters is still under development so may change
        in future releases.

    # section: reference
        .. ref_arxiv:: 1 1106.5458

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis (:class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`):
                The measurement
                :class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`
                to use for tomographic process reconstruction.
            preparation_basis (:class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`):
                The preparation
                :class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`
                to use for tomographic process reconstruction.
            fitter (str or Callable): The fitter function to use for reconstruction.
                rescale_psd (bool): If True rescale the fitted state to be positive-semidefinite
                (Default: True).
            fitter_options (dict): Any addition kwarg options to be supplied to the fitter
                function. For documentation of available kwargs refer to the fitter function
                documentation.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for
                additional information (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter
                have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
                or trace dim for :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            measurement_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                measurements. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic measurements.
            preparation_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                preparations. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic preparations.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            target (str or
                :class:`~qiskit.quantum_info.operators.channel.quantum_channel.QuantumChannel`
                or :class:`~qiskit.quantum_info.Operator`): Optional, Set a custom target quantum
                channel for computing the :func:`~qiskit.quantum_info.process_fidelity` of the
                fitted process against (Default: None).
            conditional_circuit_clbits (list[int]): Optional, the clbit indices in the
                source circuit to be conditioned on when reconstructing the channel.
                Enabling this will return a list of reconstrated channel components
                conditional on the values of these clbit values. The integer value of the
                conditioning clbits is stored in state analysis result extra field
                `"conditional_circuit_outcome"`.
            conditional_measurement_indices (list[int]): Optional, indices of tomography
                measurement qubits to used for conditional state reconstruction. Enabling
                this will return a list of reconstrated channel components conditioned on
                the remaining tomographic bases conditional on the basis index, and outcome
                value for these measurements. The conditional measurement basis index and
                integer value of the measurement outcome is stored in state analysis result
                extra fields `"conditional_measurement_index"` and
                `"conditional_measurement_outcome"` respectively.
            conditional_preparation_indices (list[int]): Optional, indices of tomography
                preparation qubits to used for conditional state reconstruction. Enabling
                this will return a list of reconstrated channel components conditioned on
                the remaining tomographic bases conditional on the basis index. The
                conditional preparation basis index is stored in state analysis result
                extra fields `"conditional_preparation_index"`.
        """
        options = super()._default_options()
        options.measurement_basis = PauliMeasurementBasis()
        options.preparation_basis = PauliPreparationBasis()
        return options
