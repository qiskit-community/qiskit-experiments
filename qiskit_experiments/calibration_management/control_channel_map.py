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

"""A qubit to control channel map."""

from typing import Any, Dict, List, Tuple

from qiskit.pulse import ControlChannel


class ControlChannelMap:
    """A class to help serialize control channel maps."""

    def __init__(self, control_channel_map: Dict[Tuple[int, ...], List[ControlChannel]]):
        """Setup the control channel map.

        Args:
            control_channel_map: A configuration dictionary of any control channels. The
                keys are tuples of qubits and the values are a list of ControlChannels
                that correspond to the qubits in the keys.
        """
        self._map = control_channel_map or {}

    @property
    def chan_map(self):
        """Return the qubits to control channel map."""
        return self._map

    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the mapping."""
        return {
            "class": self.__class__.__name__,
            "map": [{"key": k, "value": [chan.index for chan in v]} for k, v in self._map.items()],
        }

    @classmethod
    def from_config(cls, config: Dict) -> "ControlChannelMap":
        """Deserialize the control channel map given the input dictionary"""

        ch_map = config["map"]

        return cls(
            {tuple(item["key"]): [ControlChannel(idx) for idx in item["value"]] for item in ch_map}
        )

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        return self.config()

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "ControlChannelMap":
        """Load from JSON compatible format."""
        return cls.from_config(value)
