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

"""Test for fit options."""
from test.base import QiskitExperimentsTestCase

import numpy as np

from qiskit_experiments.curve_analysis.curve_data import FitOptions
from qiskit_experiments.exceptions import AnalysisError


class TestFitOptions(QiskitExperimentsTestCase):
    """Unittest for fit option object."""

    def test_empty(self):
        """Test if default value is automatically filled."""
        opt = FitOptions(["par0", "par1", "par2"])

        # bounds should be default to inf tuple. otherwise crashes the scipy fitter.
        ref_opts = {
            "p0": {"par0": None, "par1": None, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_dict(self):
        """Create option and fill with dictionary."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0={"par0": 0, "par1": 1, "par2": 2},
            default_bounds={"par0": (0, 1), "par1": (1, 2), "par2": (2, 3)},
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_array(self):
        """Create option and fill with array."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=[0, 1, 2],
            default_bounds=[(0, 1), (1, 2), (2, 3)],
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_override_partial_dict(self):
        """Create option and override value with partial dictionary."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_assigned_value(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_can_override_assigned_value_with_dict_access(self):
        """Test override already assigned value with direct dict access."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0["par1"] = 3
        opt.p0["par1"] = 5

        ref_opts = {
            "p0": {"par0": None, "par1": 5.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_user_option(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"], default_p0={"par1": 3})
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_set_operation(self):
        """Test if set works and duplicated entry is removed."""
        opt1 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt2 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt3 = FitOptions(["par0", "par1"], default_p0=[0, 2])

        opts = set()
        opts.add(opt1)
        opts.add(opt2)
        opts.add(opt3)

        self.assertEqual(len(opts), 2)

    def test_detect_invalid_p0(self):
        """Test if invalid p0 raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_p0=[0, 1])

    def test_detect_invalid_bounds(self):
        """Test if invalid bounds raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_bounds=[(0, 1), (1, 2)])

        with self.assertRaises(AnalysisError):
            # not min-max tuple
            FitOptions(["par0", "par1", "par2"], default_bounds=[0, 1, 2])

        with self.assertRaises(AnalysisError):
            # max-min tuple
            FitOptions(["par0", "par1", "par2"], default_bounds=[(1, 0), (2, 1), (3, 2)])

    def test_detect_invalid_key(self):
        """Test if invalid key raises Error."""
        opt = FitOptions(["par0", "par1", "par2"])

        with self.assertRaises(AnalysisError):
            opt.p0.set_if_empty(par3=3)

    def test_set_extra_options(self):
        """Add extra fitter options."""
        opt = FitOptions(
            ["par0", "par1", "par2"], default_p0=[0, 1, 2], default_bounds=[(0, 1), (1, 2), (2, 3)]
        )
        opt.add_extra_options(ex1=0, ex2=1)

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
            "ex1": 0,
            "ex2": 1,
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_complicated(self):
        """Test for realistic operations for algorithmic guess with user options."""
        user_p0 = {"par0": 1, "par1": None}
        user_bounds = {"par0": None, "par1": (-100, 100)}

        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=user_p0,
            default_bounds=user_bounds,
        )

        # similar computation in algorithmic guess

        opt.p0.set_if_empty(par0=5)  # this is ignored because user already provided initial guess
        opt.p0.set_if_empty(par1=opt.p0["par0"] * 2 + 3)  # user provided guess propagates

        opt.bounds.set_if_empty(par0=(0, 10))  # this will be set
        opt.add_extra_options(fitter="algo1")

        opt1 = opt.copy()  # copy options while keeping previous values
        opt1.p0.set_if_empty(par2=opt1.p0["par0"] + opt1.p0["par1"])

        opt2 = opt.copy()
        opt2.p0.set_if_empty(par2=opt2.p0["par0"] * 2)  # add another p2 value

        ref_opt1 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 6.0},
            "bounds": {"par0": (0.0, 10.0), "par1": (-100.0, 100.0), "par2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        ref_opt2 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 10.0), "par1": (-100.0, 100.0), "par2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        self.assertDictEqual(opt1.options, ref_opt1)
        self.assertDictEqual(opt2.options, ref_opt2)
