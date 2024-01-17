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
Tests for qiskit-experiments base test module
"""

from test.base import create_base_test_case


UnittestBase = create_base_test_case(use_testtools=False)


class TestQiskitExperimentsTestCaseWithUnittest(UnittestBase):
    """Test QiskitExperimentsTestCase behavior when not based on testtools.TestCase"""

    def test_test(self):
        """Test that a test not based on ``testtools`` can run"""
        pass
