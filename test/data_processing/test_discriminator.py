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

import numpy as np

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.data_processing import LDA

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestDiscriminator(QiskitExperimentsTestCase):
    """Tests for the discriminator."""

    def test_lda_serialization(self):
        """Test the serialization of a lda."""

        if HAS_SKLEARN:
            sk_lda = LinearDiscriminantAnalysis()
            sk_lda.fit([[-1, 0], [1, 0], [-1.1, 0], [0.9, 0.1]], [0, 1, 0, 1])

            self.assertTrue(sk_lda.predict([[1.1, 0]])[0], 1)

            lda = LDA(sk_lda)

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
