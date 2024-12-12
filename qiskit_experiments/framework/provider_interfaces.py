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
Definitions of interfaces for classes working with circuit execution

Qiskit Experiments tries to maintain the flexibility to work with multiple
providers of quantum circuit execution, like Qiskit IBM Runtime, Qiskit
Dynamics, and Qiskit Aer. These different circuit execution providers do not
follow exactly the same interface. This module provides definitions of the
subset of the interfaces that Qiskit Experiments needs in order to analyze
experiment results.
"""

from __future__ import annotations
from typing import Protocol, Union

from qiskit.result import Result
from qiskit.primitives import PrimitiveResult
from qiskit.providers import Backend, JobStatus


class BaseJob(Protocol):
    """Required interface definition of a job class as needed for experiment data"""

    def cancel(self):
        """Cancel the job"""
        raise NotImplementedError

    def job_id(self) -> str:
        """Return the ID string for the job"""
        raise NotImplementedError

    def result(self) -> Result | PrimitiveResult:
        """Return the job result data"""
        raise NotImplementedError

    def status(self) -> JobStatus | str:
        """Return the status of the job"""
        raise NotImplementedError


class ExtendedJob(BaseJob, Protocol):
    """Job interface with methods to support all of experiment data's features"""

    def backend(self) -> Backend:
        """Return the backend associated with a job"""
        raise NotImplementedError

    def error_message(self) -> str | None:
        """Returns the reason the job failed"""
        raise NotImplementedError


Job = Union[BaseJob, ExtendedJob]
"""Union type of job interfaces supported by Qiskit Experiments"""


class BaseProvider(Protocol):
    """Interface definition of a provider class as needed for experiment data"""

    def job(self, job_id: str) -> Job:
        """Retrieve a job object using its job ID

        Args:
            job_id: Job ID.

        Returns:
            The retrieved job
        """
        raise NotImplementedError


class IBMProvider(BaseProvider, Protocol):
    """Provider interface needed for supporting features like IBM Quantum

    This interface is the subset of
    :class:`~qiskit_ibm_runtime.QiskitRuntimeService` needed for all features
    of Qiskit Experiments. Another provider could implement this interface to
    support these features as well.
    """

    def active_account(self) -> dict[str, str] | None:
        """Return the IBM Quantum account information currently in use

        This method returns the current account information in a dictionary
        format. It is used to copy the credentials for use with
        ``qiskit-ibm-experiment`` without requiring specifying the credentials
        for the provider and ``qiskit-ibm-experiment`` separately
        It should include ``"url"`` and ``"token"`` as keys for the
        authentication to work.

        Returns:
            A dictionary with information about the account currently in the session.
        """
        raise NotImplementedError


Provider = Union[BaseProvider, IBMProvider]
"""Union type of provider interfaces supported by Qiskit Experiments"""
