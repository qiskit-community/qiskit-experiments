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

from matplotlib.figure import Figure as MatplotlibFigure


class FigureData:
    """A plot data container.

    .. note::
        Raw figure data can be accessed through the :attr:`.FigureData.figure` attribute.

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

    def __eq__(self, value):
        """Test equality between two instances of FigureData."""
        return vars(self) == vars(value)

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
            return self.figure.decode("utf-8")
        return None


FigureType = Union[str, bytes, MatplotlibFigure, FigureData]
