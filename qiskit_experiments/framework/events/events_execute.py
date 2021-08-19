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
Execute events.
"""

from qiskit import assemble
from qiskit.providers.basebackend import BaseBackend as LegacyBackend


def backend_run(experiment, backend, circuits, run_options, **kwargs):

    if isinstance(backend, LegacyBackend):
        qobj = assemble(circuits, backend=backend, **run_options)
        job = backend.run(qobj)
    else:
        job = backend.run(circuits, **run_options)

    return {"job": job}
