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

"""Discriminator wrappers to make discriminators serializable.."""

from abc import abstractmethod
from typing import Any, Dict, List


class BaseDiscriminator:
    """An abstract base class for serializable discriminators.

    ``BaseDiscriminator``s are used in the :class:`.Discriminator` data action nodes.

    This class allows developers to implement their own discriminators or wrap discriminators
    from external libraries which therefore ensures that the discriminator fits in
    the data processing chain. This class defines an interface for discriminator objects.
    Subclasses must implement the following methods:
    - :meth:`predict`: called in the :class:`.Discriminator` data-action class to predict
      labels from the input level-one data.
    - :meth:`config`: produces the config file to serialize and deserialize the discriminator.
    - :meth:`is_trained`: indicates if the discriminator is trained, i.e., fit to training data.
    """

    @abstractmethod
    def predict(self, data: List):
        """The function used to predict the labels of the data."""

    @property
    def discriminator(self) -> Any:
        """Return the discriminator object that is wrapped.

        Sub-classes may not need to implement this method but can chose to
        if they are wrapping an object capable of discrimination.
        """
        return None

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return the configuration of the discriminator."""

    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if this discriminator has been trained on data."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseDiscriminator":
        """Create a discriminator from the configuration."""

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        return self.config()

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "BaseDiscriminator":
        """Load from JSON compatible format."""
        return cls.from_config(value)
