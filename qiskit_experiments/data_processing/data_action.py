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

"""Defines the steps that can be used to analyse data."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Any

import numpy as np

from qiskit_experiments.framework.store_init_args import StoreInitArgs
from qiskit_experiments.framework import Options


class DataAction(ABC, StoreInitArgs):
    """Abstract action done on measured data to process it.

    Each subclass of DataAction must define the way it formats, validates and processes data.
    """

    def __init__(self, validate: bool = True):
        """Create new node.

        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        self._validate = validate

    @abstractmethod
    def _process(self, data: np.ndarray) -> np.ndarray:
        """Applies the data processing step to the data.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.
                If the elements are ufloat objects consisting of a nominal value and
                a standard error, then the error propagation is automatically computed.

        Returns:
            The processed data.
        """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format and validate the input.

        Check that the given data has the correct structure. This method may
        additionally change the data type, e.g. converting a list to a numpy array.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been validated and formatted.
        """
        return data

    def __json_encode__(self) -> Dict[str, Any]:
        """Return the config dict for this node."""
        return dict(
            cls=type(self),
            args=tuple(getattr(self, "__init_args__", OrderedDict()).values()),
            kwargs=dict(getattr(self, "__init_kwargs__", OrderedDict())),
        )

    @classmethod
    def __json_decode__(cls, config: Dict[str, Any]) -> "DataAction":
        """Initialize a node from config dict."""
        init_args = config.get("args", tuple())
        init_kwargs = config.get("kwargs", dict())

        return cls(*init_args, **init_kwargs)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call the data action of this node on the data.

        Args:
            data: A numpy array with arbitrary dtype. If the elements are ufloat objects
                consisting of a nominal value and a standard error, then the error propagation
                is done automatically.

        Returns:
            The processed data.
        """
        return self._process(self._format_data(data))

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate={self._validate})"


class TrainableDataAction(DataAction):
    """A base class for data actions that need training."""

    def __init__(self, validate: bool = True):
        """Create new node.

        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate=validate)
        self._parameters = self._default_parameters()

    @classmethod
    def _default_parameters(cls) -> Options:
        """Set default node parameters."""
        return Options()

    @property
    def parameters(self) -> Options:
        """Returns the trained node parameters."""
        return self._parameters

    def set_parameters(self, **fields):
        """Set parameters from training."""
        for field in fields:
            if not hasattr(self._parameters, field):
                raise AttributeError(
                    f"Parameters field {field} is not valid for {type(self).__name__}."
                )
        self._parameters.update_options(**fields)

    @property
    def is_trained(self) -> bool:
        """Return False if the DataAction needs to be trained.

        Return:
            True if the data action has been trained.
        """
        return all(p is not None for p in self.parameters.__dict__.values())

    @abstractmethod
    def train(self, data: np.ndarray):
        """Train a DataAction.

        Certain data processing nodes, such as a SVD, require data to first train.

        Args:
            data: A data array for training. This is a single numpy array containing
                all circuit results input to the data processor :meth:`~qiskit_experiments.\
                data_processing.data_processor.DataProcessor#train` method.
        """

    def __json_encode__(self) -> Dict[str, Any]:
        """Return the config dict for this node."""
        config = super().__json_encode__()
        config["params"] = self.parameters.__dict__

        return config

    @classmethod
    def __json_decode__(cls, config: Dict[str, Any]) -> "TrainableDataAction":
        """Initialize a node from config dict."""
        params = config.pop("params", dict())
        instance = super().__json_decode__(config)

        # pylint: disable=no-member
        instance.set_parameters(**params)

        return instance

    def __repr__(self):
        """String representation of the node."""
        options_str = f"validate={self._validate}"
        for pname, pval in self.parameters.__dict__.items():
            options_str += f", {pname}={pval}"
        return f"{self.__class__.__name__}({options_str})"
