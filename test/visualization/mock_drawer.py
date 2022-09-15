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
"""
Mock drawer for testing.
"""

from typing import Optional, Sequence

from qiskit_experiments.visualization import BaseDrawer, PlotStyle


class MockDrawer(BaseDrawer):
    """Mock drawer for visualization tests.

    Most methods of this class do nothing.
    """

    @property
    def figure(self):
        """Does nothing."""
        pass

    @classmethod
    def _default_style(cls) -> PlotStyle:
        """Default style.

        Style Param:
            overwrite_param: A test style parameter to be overwritten by a test.
        """
        style = super()._default_style()
        style.overwrite_param = "overwrite_param"
        return style

    def initialize_canvas(self):
        """Does nothing."""
        pass

    def format_canvas(self):
        """Does nothing."""
        pass

    def draw_raw_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Does nothing."""
        pass

    def draw_formatted_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Does nothing."""
        pass

    def draw_line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Does nothing."""
        pass

    def draw_confidence_interval(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Does nothing."""
        pass

    def draw_report(
        self,
        description: str,
        **options,
    ):
        """Does nothing."""
        pass
