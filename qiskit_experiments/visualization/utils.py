# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Utilities for visualization."""
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit.exceptions import QiskitError

ExtentTuple = Tuple[float, float, float, float]


class DataExtentCalculator:
    """A class to handle computing the extent of two-dimensional data.

    This class makes computing the extent of two-dimensional data easier, especially for
    adding images to plots where the extent of the axes (i.e., axis limits) are not
    known prior to figure generation. An instance of this class can be used to compute
    an extent tuple which is larger than the registered data, and thus has a margin that
    can exceed the axis-limits or be used as the axis limits without cropping additional
    data-points from a figure. An extent tuple ``(x_min, x_max, y_min, y_max)`` contains
    the minimum and maximum values for the X and Y dimensions.

    Data is registered with a :class:`DataExtentCalculator` so that the computed extent
    covers all values in the data array. The extent tuple is computed as follows:

    1. The maximum and minimum values for input data is stored whenever new data
       arrays are registered. This is the data-extent: the minimum-area bounding box
       that contains all registered data.
    2. The data-extent is enlarged/shrunk by scaling its width and height by
       :attr:`multiplier`.
    3. If :attr:`aspect_ratio` is not ``None``, the scaled extent tuple is extended
       in one of the dimensions so that the output extent tuple is larger and the
       target aspect ratio is achieved.
    """

    def __init__(self, multiplier: float = 1.0, aspect_ratio: Optional[float] = 1.0):
        """Create an extent calculator.

        Args:
            multiplier: The factor by which to scale the extent of the data when
                computing the output extent tuple. Defaults to 1.0.
            aspect_ratio: An optional target aspect ratio for the output extent tuple.
                If None, the extent tuple is not extended to achieve a given aspect
                ratio. Defaults to None.
        """
        self._multiplier = multiplier
        self._aspect_ratio = aspect_ratio
        self._data_extent = np.asarray([[np.inf, -np.inf], [np.inf, -np.inf]])

    @property
    def multiplier(self) -> float:
        """The multiplier by which to scale the data-extent.

        Returns:
            The multiplier for the computed extent.
        """
        return self._multiplier

    @property
    def aspect_ratio(self) -> Optional[float]:
        """The target aspect ratio.

        If None, the :class:`DataExtentCalculator` instance will not modify the computed
        extent tuple to achieve a given aspect ratio; instead only scale by
        :attr:`multiplier`.

        Returns:
            The target aspect ratio of the computed extent tuple.
        """
        return self._aspect_ratio

    def register_data(self, data: Union[List, np.ndarray], dim: Optional[int] = None):
        r"""Register data to modify the resulting extent tuple.

        Args:
            data: Array or list of data values to use when calculating the extent tuple.
                If a list is given, it is converted into an array using
                :meth:`numpy.asarray`. If the array has the shape ``(m, 2)``, then there
                are ``m`` values in two dimensions (being the second dimension of
                ``data``). If the array has the shape ``(m,)`` or ``(m, 1)``, then
                ``dim`` must be set to the index of the dimension for which this data is
                associated.
            dim: Optional dimension index if a one-dimensional array is provided (i.e.,
                `X=0` and `Y=1`). If None, then ``data`` must have the shape ``(m, 2)``.
                Defaults to None.

        Raises:
            QiskitError: If the data is not two-dimensional and ``dim`` is not set.
            QiskitError: If the data does not contain one-dimensional values when
            ``dim`` is set.
            QiskitError: If ``dim`` is not an index for two-dimensions: i.e.,
            :math:`0\leq{}\text{dim}<2`.
        """
        data = np.asarray(data)
        if dim is None and (len(data.shape) != 2 or (len(data.shape) == 2 and data.shape[1] != 2)):
            raise QiskitError(
                "Data must contain two-dimensional values. Expected shape `(m,2)` but got "
                f"{data.shape} instead."
            )
        if dim is not None and (
            len(data.shape) > 2 or (len(data.shape) == 2 and data.shape[1] != 1)
        ):
            raise QiskitError(
                "Data must contain one-dimensional values if `dim` is set. Expected shape of `(m,1)` "
                f"or `(m,)` but got {data.shape} instead."
            )
        if dim is not None and dim >= 2:
            raise QiskitError(
                f"Dim must be a two-dimensional dimension index (0<=dim<2), got {dim} instead."
            )

        if dim is not None:
            _data_extent = [
                min(np.min(data), self._data_extent[dim, 0]),
                max(np.max(data), self._data_extent[dim, 1]),
            ]
            self._data_extent[dim] = _data_extent
        else:
            for i_dim in range(2):
                _data_extent = [
                    min(np.min(data[..., i_dim]), self._data_extent[i_dim, 0]),
                    max(np.max(data[..., i_dim]), self._data_extent[i_dim, 1]),
                ]
                self._data_extent[i_dim, :] = _data_extent

    @classmethod
    def _range(cls, extent: np.ndarray) -> np.ndarray:
        """Compute the range of the extent array.

        Args:
            extent: The extent array for the range.

        Returns:
            The array ``[x_range, y_range]`` where ``x_range`` and ``y_range`` are the
            ranges for their respective dimensions.
        """
        return np.diff(
            extent,
            axis=1,
        ).flatten()

    @classmethod
    def _midpoint(cls, extent: np.ndarray) -> np.ndarray:
        """Compute the midpoint of the extent array.

        Args:
            extent: The extent array for the midpoint.

        Returns:
            The array ``[x, y]`` where ``x`` and ``y`` are the midpoints for their
            respective dimensions.
        """
        return np.mean(
            extent,
            axis=1,
        )

    @classmethod
    def _extent_from_range_midpoint(
        cls, extent_range: np.ndarray, midpoint: np.ndarray
    ) -> np.ndarray:
        """Compute an extent array from range and midpoint arrays.

        Args:
            extent_range: The extent range to use.
            midpoint: The midpoint of the extent.

        Returns:
            An extent array with range and midpoint corresponding to ``extent_range``
            and ``midpoint``.
        """
        radii = extent_range.flatten() / 2
        new_extent = np.zeros((2, 2))
        new_extent[:, 0] = midpoint - radii
        new_extent[:, 1] = midpoint + radii
        return new_extent

    @classmethod
    def _extent_with_aspect_ratio(
        cls, extent: np.ndarray, target_aspect_ratio: Optional[float]
    ) -> np.ndarray:
        """Expand ``extent`` to have an aspect ratio defined by ``target_aspect_ratio``.

        This method extends ``extent`` along one of the two dimensions so that its
        aspect ratio is equal to ``target_aspect_ratio``. This is done by computing the
        ratio of the actual and target aspect ratios, computing the dimension along
        which to extend ``extent``, and recomputing the extent array based on scaled
        ranges to achieve the target aspect ratio. If the target ratio is ``None``,
        nothing is done. If extended, the region defined by the output array is always
        larger than the input array.

        Args:
            extent: The extent array to expand.
            target_aspect_ratio: Optional target aspect ratio, being
                :math:`\text{width}/\text{height}` for the extent array. If None,
                ``extent`` is not extended. Defaults to None.

        Returns:
            ``extent`` extended to have an aspect ratio defined by
            ``target_aspect_ratio``.
        """
        if target_aspect_ratio is None:
            return extent

        # Compute ratio coefficient
        _range = cls._range(extent)
        ratio = _range[0] / _range[1]
        ratio_coefficient = target_aspect_ratio / ratio

        _new_range = _range
        # Handle three cases of aspect ratios
        if ratio_coefficient == 1.0:
            return extent
        elif ratio_coefficient < 1:
            _new_range[1] = (1 / ratio_coefficient) * _range[1]
        else:
            _new_range[0] = ratio_coefficient * _range[0]

        return cls._extent_from_range_midpoint(_new_range, cls._midpoint(extent))

    @classmethod
    def _scaled_extent(cls, extent: np.ndarray, multiplier: float) -> np.ndarray:
        _range = cls._range(extent)
        midpoint = cls._midpoint(extent)
        new_range = multiplier * _range
        return cls._extent_from_range_midpoint(new_range, midpoint)

    def extent(self) -> ExtentTuple:
        """An extent array for the registered data, multiplier, and aspect ratio.

        Raises:
            QiskitError: If the resulting extent tuple is not finite. This can occur if
            no data was registered before calling :meth:`extent`.

        Returns:
            The extent tuple for the registered data, scaled by ``multiplier``, and then
            extended to achieve the set aspect ratio.
        """
        if not np.all(np.isfinite(self._data_extent)):
            is_infinite = np.argwhere(np.invert(np.isfinite(self._data_extent)))
            raise QiskitError(
                "Encountered non-finite extent. The following dimensions have non-finite "
                f"values: {is_infinite}."
            )

        return tuple(
            self._extent_with_aspect_ratio(
                self._scaled_extent(self._data_extent, self._multiplier), self._aspect_ratio
            ).flatten()
        )
