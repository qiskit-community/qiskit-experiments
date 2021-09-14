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

"""A collection of common operation callback in execution chain."""


from qiskit.providers import Backend
from qiskit.test.mock import FakeBackend

from .base_experiment import BaseExperiment


def apply_delay_validation(experiment: BaseExperiment, backend: Backend):

    if_simulator = getattr(backend.configuration(), "simulator", False)

    if not if_simulator and not isinstance(backend, FakeBackend):
        timing_constraints = getattr(
            experiment.transpile_options.__dict__, "timing_constraints", {}
        )

        # alignment=16 is IBM standard. Will be soon provided by IBM providers.
        # Then, this configuration can be removed.
        timing_constraints["acquire_alignment"] = getattr(
            timing_constraints, "acquire_alignment", 16
        )

        scheduling_method = getattr(
            experiment.transpile_options.__dict__, "scheduling_method", "alap"
        )
        experiment.set_transpile_options(
            timing_constraints=timing_constraints, scheduling_method=scheduling_method
        )
