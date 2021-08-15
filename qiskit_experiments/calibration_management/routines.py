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

"""A collections of experiment wrapper functions to facilitate calibration."""

from typing import Tuple, Union
import numpy as np

from qiskit.circuit import Parameter

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library.calibration.rabi import Rabi
from qiskit_experiments.library.calibration.drag import DragCal
from qiskit_experiments.library.calibration.fine_amplitude import FineXAmplitude, FineSXAmplitude
from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.calibration_management.update_library import Amplitude, Drag, Frequency
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations


def spectroscopy(
    calibrations: BackendCalibrations,
    qubits: Union[int, Tuple[int]],
    backend,
    freq_range: float = 15e6,
    experiment_options=None,
):
    """Wrapper function to call spectroscopy experiments.

    Args:
        calibrations: An instance of :class:`BackendCalibrations` which holds the schedules and
            parameters that will be updated.
        qubits: The qubits to calibrate.
        backend: The backend on which to run.
        freq_range: The experiment will scan frequencies ranging from the default frequency in
            the backend instance less freq_range to the default frequency plus the given
            freq_range.
        experiment_options: Options to provide to the experiment. These are the options that the
            :class:`QubitSpectroscopy` experiment takes.

    Returns:
        The data from the parallel experiment that will be run.
    """

    if isinstance(qubits, int):
        qubits = [qubits]

    # 1. Setup the experiment.
    specs = []
    for qubit in qubits:
        freq01_estimate = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01_estimate - freq_range, freq01_estimate + freq_range, 51)

        spec = QubitSpectroscopy(qubit, frequencies)

        if experiment_options is not None:
            spec.set_experiment_options(**experiment_options)

        specs.append(spec)

    spec = ParallelExperiment(specs)

    # 2. Run the experiment.
    spec_data = spec.run(backend).block_for_results()

    # 3. Update the calibrations.
    for idx in range(len(qubits)):
        data = spec_data.component_experiment_data(idx)
        Frequency.update(calibrations, data)


def roughamp(
    calibrations: Calibrations,
    qubits: Union[int, Tuple[int]],
    backend,
    schedule_name: str = "x",
    half_angle_schedule_name="sx",
    experiment_options=None,
) -> ExperimentData:
    """Run the Rabi amplitude calibration.

    Args:
        calibrations: An instance of :class:`Calibrations` which holds the schedules and parameters
            that will be updated.
        qubits: The qubits to calibrate.
        backend: The backend on which the experiment will be run.
        schedule_name: The name of the schedule as found in the cals for which to run the
            rough amplitude calibration.
        half_angle_schedule_name: Name of the half angle schedule to update.
        experiment_options: Options to provide to the experiment. These are the options that the
            :class:`Rabi` experiment takes.

    Returns:
        The data from the parallel experiment that will be run.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    # 1. Setup the experiment.
    rabis = []
    for qubit in qubits:
        rabi = Rabi(qubit)

        sched = calibrations.get_schedule(
            schedule_name, qubit, assign_params={"amp": Parameter("amp")}
        )

        rabi.set_experiment_options(schedule=sched)

        if experiment_options is not None:
            rabi.set_experiment_options(**experiment_options)

        rabis.append(rabi)

    rabi = ParallelExperiment(rabis)

    # 2. Run the experiment.
    rabi_data = rabi.run(backend).block_for_results()

    # 3. Update the calibrations.
    angles_schedules = [(np.pi, "amp", schedule_name)]
    if half_angle_schedule_name is not None:
        angles_schedules += [(np.pi / 2, "amp", half_angle_schedule_name)]

    for idx in range(len(qubits)):
        data = rabi_data.component_experiment_data(idx)
        Amplitude.update(calibrations, data, angles_schedules=angles_schedules)

    return rabi_data


def roughdrag(
    calibrations: Calibrations,
    qubits: Union[int, Tuple[int]],
    backend,
    schedule_name: str = "x",
    experiment_options=None,
) -> ExperimentData:
    """Run the rough Drag calibration.

    Args:
        calibrations: An instance of :class:`Calibrations` which holds the schedules and parameters
            that will be updated.
        qubits: The qubits to calibrate.
        backend: The backend on which the experiment will be run.
        schedule_name: The name of the schedule as found in the cals for which to run the
            DRAG calibration.
        experiment_options: Options to provide to the experiment. These are the options that the
            :class:`DragCal` experiment takes.

    Returns:
        The data from the parallel experiment that will be run.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    # 1. Setup the experiments
    drags = []
    for qubit in qubits:
        drag = DragCal(qubit)
        sched = calibrations.get_schedule(schedule_name, qubit, assign_params={"β": Parameter("β")})

        drag.set_experiment_options(rp=sched)

        if experiment_options is not None:
            drag.set_experiment_options(**experiment_options)

        drags.append(drag)

    drag = ParallelExperiment(drags)

    # 2. Run the experiment
    drag_data = drag.run(backend).block_for_results()

    # 3. Update the calibrations
    for idx in range(len(qubits)):
        data = drag_data.component_experiment_data(idx)
        Drag.update(calibrations, data, parameter="β", schedule=schedule_name)

    return drag_data


def fineamp(
    calibrations: Calibrations,
    qubits: Union[int, Tuple[int]],
    backend,
    x_schedule_name: str = "x",
    sx_schedule_name: str = "sx",
    angle: float = np.pi,
    experiment_options=None,
):
    """Wrapper function to perform fine amplitude calibration on pi or pi-half pulses.

    Args:
        calibrations: An instance of :class:`Calibrations` which holds the schedules and parameters
            that will be updated.
        qubits: The qubits to calibrate.
        backend: The backend on which the experiment will be run.
        x_schedule_name: The name of the x schedule as found in the calibrations for which to run
            the fine amplitude calibration.
        sx_schedule_name: The name of the square root x schedule as found in the calibrations. This
            is the schedule that will be calibrated if angle is np.pi/2. If angle is np.pi then
            this is the schedule that will be used to move to the equator of the Bloch sphere.
        angle: The angle for which to run the calibrations. Currently, only np.pi and np.pi/2 are
            supported.
        experiment_options: Options to provide to the experiment. These are the options that the
            :class:`FineAmplitude` experiment takes.

    Returns:
        The data from the parallel experiment that will be run.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    # 1. Construct the experiments
    experiment = None
    cal_schedule_name = None
    if np.allclose(angle, np.pi):
        experiment = FineXAmplitude
        cal_schedule_name = x_schedule_name

    if np.allclose(angle, np.pi / 2):
        experiment = FineSXAmplitude
        cal_schedule_name = sx_schedule_name

    if experiment is None:
        raise CalibrationError(f"fineamp only supports pi and pi-half angles. Received {angle}.")

    fineamps = []
    for qubit in qubits:
        fine_amp = experiment(qubit)

        fine_amp.set_experiment_options(
            schedule=calibrations.get_schedule(cal_schedule_name, qubit),
            sx_schedule=calibrations.get_schedule(sx_schedule_name, qubit),
        )

        if experiment_options is not None:
            fine_amp.set_experiment_options(**experiment_options)

        fineamps.append(fine_amp)

    fine_amplitude = ParallelExperiment(fineamps)

    # 2. Run the experiment
    fine_amp_data = fine_amplitude.run(backend).block_for_results()

    # 3. Update the calibrations
    for idx in range(len(qubits)):
        data = fine_amp_data.component_experiment_data(idx)
        Amplitude.update(calibrations, data, angles_schedules=[(angle, "amp", cal_schedule_name)])

    return fine_amp_data
