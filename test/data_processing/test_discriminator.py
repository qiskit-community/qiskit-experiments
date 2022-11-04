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

from qiskit_experiments.data_processing import SkLDA, SkCLF

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def requires_sklearn(func):
    """Decorator to check for SKLearn."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_SKLEARN:
            raise SkipTest("SKLearn is required for test.")

        func(*args, **kwargs)

    return wrapper


class TestDiscriminator(QiskitExperimentsTestCase):
    """Tests for the discriminator."""

    @requires_sklearn
    def test_lda_serialization(self):
        """Test the serialization of a lda."""

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

        self.assertRoundTripSerializable(lda, check_lda)

    @requires_sklearn
    def test_skclf_from_config(self):
        """Test building SkCLF from configuration."""

        sk_clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3)
        clf = SkCLF(sk_clf)

        clf.fit([[-1, 0], [1, 0], [-1.1, 0], [0.9, 0.1]], [0, 1, 0, 1])

        self.assertTrue(clf.is_trained())
        self.assertTrue(clf.predict([[1.1, 0]])[0], 1)

        config = clf.config()

        clf_from_config = SkCLF.from_config(config)

        self.assertTrue(clf_from_config.predict([[1.1, 0]])[0], 1)

    @requires_sklearn
    def test_skclf_serialization(self):
        """Test the serialization of a SkCLF."""

        sk_clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3)
        sk_clf = make_pipeline(StandardScaler(), sk_clf)
        sk_clf.fit([[-1, 0], [1, 0], [-1.1, 0], [0.9, 0.1]], [0, 1, 0, 1])

        self.assertTrue(sk_clf.predict([[1.1, 0]])[0], 1)

        clf = SkCLF(sk_clf)

        self.assertTrue(clf.is_trained())
        self.assertTrue(clf.predict([[1.1, 0]])[0], 1)

        def check_clf(clf1, clf2):
            test_data = [[1.1, 0], [0.1, 0], [-2, 0]]

            clf1_y = clf1.predict(test_data)
            clf2_y = clf2.predict(test_data)

            if len(clf1_y) != len(clf2_y):
                return False

            for idx, y_val1 in enumerate(clf1_y):
                if clf2_y[idx] != y_val1:
                    return False

            for attribute in clf1.attributes:
                if not np.allclose(
                        getattr(clf1.discriminator, attribute, np.array([])),
                        getattr(clf2.discriminator, attribute, np.array([])),
                ):
                    return False

            return True

        self.assertRoundTripSerializable(clf, check_clf)
