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

"""Tests for the serializable discriminator objects."""

from test.base import QiskitExperimentsTestCase
from functools import wraps
from unittest import SkipTest
import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_experiments.data_processing import SkLDA, SkQDA
from qiskit_experiments.framework.package_deps import HAS_SKLEARN


def requires_sklearn(func):
    """Decorator to check for SKLearn."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            HAS_SKLEARN.require_now("SKLearn discriminator testing")
        except MissingOptionalLibraryError as exc:
            raise SkipTest("SKLearn is required for test.") from exc

        func(*args, **kwargs)

    return wrapper


class TestDiscriminator(QiskitExperimentsTestCase):
    """Tests for the discriminator."""

    @requires_sklearn
    def test_lda_serialization(self):
        """Test the serialization of a lda."""

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        sk_lda = LinearDiscriminantAnalysis()
        sk_lda.fit([[-1, 0], [1, 0], [-1.1, 0], [0.9, 0.1]], [0, 1, 0, 1])

        self.assertTrue(sk_lda.predict([[1.1, 0]])[0], 1)

        lda = SkLDA(sk_lda)

        self.assertTrue(lda.is_trained())
        self.assertTrue(lda.predict([[1.1, 0]])[0], 1)

        def check_lda(lda1, lda2):
            test_data = [[1.1, 0], [0.1, 0], [-2, 0]]

            lda1_y = lda1.predict(test_data)
            lda2_y = lda2.predict(test_data)

            if len(lda1_y) != len(lda2_y):
                return False

            for idx, y_val1 in enumerate(lda1_y):
                if lda2_y[idx] != y_val1:
                    return False

            for attribute in lda1.attributes:
                if not np.allclose(
                    getattr(lda1.discriminator, attribute, np.array([])),
                    getattr(lda2.discriminator, attribute, np.array([])),
                ):
                    return False

            return True

        self.assertRoundTripSerializable(lda, check_func=check_lda)

    @requires_sklearn
    def test_qda_serialization(self):
        """Test the serialization of a qda."""

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        sk_qda = QuadraticDiscriminantAnalysis()
        sk_qda.fit([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], [0, 0, 0, 1, 1, 1])

        self.assertTrue(sk_qda.predict([[1.1, 3]])[0], 1)

        qda = SkQDA(sk_qda)

        self.assertTrue(qda.is_trained())
        self.assertTrue(qda.predict([[1.1, 3]])[0], 1)

        def check_qda(qda1, qda2):
            test_data = [[1.1, 0], [0.1, 0], [-2, 0]]

            qda1_y = qda1.predict(test_data)
            qda2_y = qda2.predict(test_data)

            if len(qda1_y) != len(qda2_y):
                return False

            for idx, y_val1 in enumerate(qda1_y):
                if qda2_y[idx] != y_val1:
                    return False

            for attribute in qda1.attributes:
                if not np.allclose(
                    getattr(qda1.discriminator, attribute, np.array([])),
                    getattr(qda2.discriminator, attribute, np.array([])),
                ):
                    return False

            return True

        self.assertRoundTripSerializable(qda, check_func=check_qda)
