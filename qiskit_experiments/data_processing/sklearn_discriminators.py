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

"""Discriminators that wrap SKLearn."""

from typing import Any, List, Dict, TYPE_CHECKING

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.framework.package_deps import HAS_SKLEARN

if TYPE_CHECKING:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )


class SkLDA(BaseDiscriminator):
    """A wrapper for the scikit-learn linear discriminant analysis.

    .. note::
        This class requires that scikit-learn is installed.
    """

    def __init__(self, lda: "LinearDiscriminantAnalysis"):
        """
        Args:
            lda: The sklearn linear discriminant analysis. This may be a trained or an
                untrained discriminator.

        Raises:
            DataProcessorError: If SKlearn could not be imported.
        """
        self._lda = lda
        self.attributes = [
            "coef_",
            "intercept_",
            "covariance_",
            "explained_variance_ratio_",
            "means_",
            "priors_",
            "scalings_",
            "xbar_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
        ]

    @property
    def discriminator(self) -> Any:
        """Return then SKLearn object."""
        return self._lda

    def is_trained(self) -> bool:
        """Return True if the discriminator has been trained on data."""
        return not getattr(self._lda, "classes_", None) is None

    def predict(self, data: List):
        """Wrap the predict method of the LDA."""
        return self._lda.predict(data)

    def fit(self, data: List, labels: List):
        """Fit the LDA.

        Args:
            data: The independent data.
            labels: The labels corresponding to data.
        """
        self._lda.fit(data, labels)

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the LDA."""
        attr_conf = {attr: getattr(self._lda, attr, None) for attr in self.attributes}
        return {"params": self._lda.get_params(), "attributes": attr_conf}

    @classmethod
    @HAS_SKLEARN.require_in_call
    def from_config(cls, config: Dict[str, Any]) -> "SkLDA":
        """Deserialize from an object."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis()
        lda.set_params(**config["params"])

        for name, value in config["attributes"].items():
            if value is not None:
                setattr(lda, name, value)

        return SkLDA(lda)


class SkQDA(BaseDiscriminator):
    """A wrapper for the SKlearn quadratic discriminant analysis.

    .. note::
        This class requires that scikit-learn is installed.
    """

    def __init__(self, qda: "QuadraticDiscriminantAnalysis"):
        """
        Args:
            qda: The sklearn quadratic discriminant analysis. This may be a trained or an
                untrained discriminator.

        Raises:
            DataProcessorError: If SKlearn could not be imported.
        """
        self._qda = qda
        self.attributes = [
            "coef_",
            "intercept_",
            "covariance_",
            "explained_variance_ratio_",
            "means_",
            "priors_",
            "scalings_",
            "xbar_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
            "rotations_",
        ]

    @property
    def discriminator(self) -> Any:
        """Return then SKLearn object."""
        return self._qda

    def is_trained(self) -> bool:
        """Return True if the discriminator has been trained on data."""
        return not getattr(self._qda, "classes_", None) is None

    def predict(self, data: List):
        """Wrap the predict method of the QDA."""
        return self._qda.predict(data)

    def fit(self, data: List, labels: List):
        """Fit the QDA.

        Args:
            data: The independent data.
            labels: The labels corresponding to data.
        """
        self._qda.fit(data, labels)

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the QDA."""
        attr_conf = {attr: getattr(self._qda, attr, None) for attr in self.attributes}
        return {"params": self._qda.get_params(), "attributes": attr_conf}

    @classmethod
    @HAS_SKLEARN.require_in_call
    def from_config(cls, config: Dict[str, Any]) -> "SkQDA":
        """Deserialize from an object."""
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        qda = QuadraticDiscriminantAnalysis()
        qda.set_params(**config["params"])

        for name, value in config["attributes"].items():
            if value is not None:
                setattr(qda, name, value)

        return SkQDA(qda)
