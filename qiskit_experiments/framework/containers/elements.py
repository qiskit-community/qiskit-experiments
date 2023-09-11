# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Container of experiment data components."""

from __future__ import annotations

import copy
import io
from typing import Dict, Optional, Union, Any

import numpy as np

from matplotlib.figure import Figure as MatplotlibFigure
from qiskit.result import Counts


class FigureData:
    """A plot data container.

    .. notes::
        Raw figure data can be accessed through the :attr:`.figure` attribute.

    """

    def __init__(
        self,
        figure,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Creates a new figure data object.

        Args:
            figure: The raw figure itself. Can be SVG or matplotlib.Figure.
            name: The name of the figure.
            metadata: Any metadata to be stored with the figure.
        """
        self.figure = figure
        self._name = name
        self.metadata = metadata or {}

    # name is read only
    @property
    def name(self) -> str:
        """The name of the figure"""
        return self._name

    @property
    def metadata(self) -> dict:
        """The metadata dictionary stored with the figure"""
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata: dict):
        """Set the metadata to new value; must be a dictionary"""
        if not isinstance(new_metadata, dict):
            raise ValueError("figure metadata must be a dictionary")
        self._metadata = new_metadata

    def copy(self, new_name: Optional[str] = None):
        """Creates a copy of the figure data"""
        name = new_name or self.name
        return FigureData(figure=self.figure, name=name, metadata=copy.deepcopy(self.metadata))

    def __json_encode__(self) -> Dict[str, Any]:
        """Return the json representation of the figure data"""
        return {"figure": self.figure, "name": self.name, "metadata": self.metadata}

    @classmethod
    def __json_decode__(cls, args: Dict[str, Any]) -> "FigureData":
        """Initialize a figure data from the json representation"""
        return cls(**args)

    def _repr_png_(self):
        if isinstance(self.figure, MatplotlibFigure):
            b = io.BytesIO()
            self.figure.savefig(b, format="png", bbox_inches="tight")
            png = b.getvalue()
            return png
        else:
            return None

    def _repr_svg_(self):
        if isinstance(self.figure, str):
            return self.figure
        if isinstance(self.figure, bytes):
            return str(self.figure)
        return None


class CanonicalResult(dict):
    """A canonical result representation of experiment circuit execution."""

    def __init__(
        self,
        job_id: str | None = None,
        counts: dict | Counts | None = None,
        memory: np.ndarray | list | None = None,
        metadata: dict | None = None,
        shots: int | None = None,
        meas_level: int | None = None,
        meas_return: str | None = None,
        creg_sizes: list | None = None,
        memory_slots: int | None = None,
    ):
        """Create new circuit result object.

        Args:
            job_id: ID for experiment job.
            counts: Discriminated count dictionary.
            memory: Raw measurement data.
            metadata: Experiment metadata.
            shots: Number of shots for this circuit execution.
            meas_level: Measurement level for this execution.
            meas_return: Return data format for this execution.
            creg_sizes: Size of each classical registers.
            memory_slots: Total size of memory slots.
        """
        if counts and not isinstance(counts, Counts):
            counts = Counts(counts, creg_sizes=creg_sizes, memory_slots=memory_slots)

        super().__init__(
            job_id=job_id,
            counts=counts,
            memory=memory,
            metadata=metadata,
            shots=shots,
            meas_level=meas_level,
            meas_return=meas_return,
            creg_sizes=creg_sizes,
            memory_slots=memory_slots,
        )

    def __json_encode__(self) -> dict[str, Any]:
        return {
            "job_id": self["job_id"],
            "counts": self["counts"],
            "memory": self["memory"],
            "metadata": self["metadata"],
            "shots": self["shots"],
            "meas_level": self["meas_level"],
            "meas_return": self["meas_return"],
            "creg_sizes": self["creg_sizes"],
            "memory_slots": self["memory_slots"],
        }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "CanonicalResult":
        return cls(**value)


_FigureT = Union[str, bytes, MatplotlibFigure, FigureData]
