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

"""Class to store the results of a calibration experiments."""

import copy
from datetime import datetime
import dataclasses
from typing import Tuple, Union, List, Optional, Type
import pandas as pd

from qiskit.circuit import Gate
from qiskit import QuantumCircuit
from qiskit.pulse import Schedule, DriveChannel, ControlChannel, MeasureChannel
from qiskit.pulse.channels import PulseChannel
from qiskit.circuit import Parameter
from .exceptions import CalibrationError
from .parameter_value import ParameterValue


class CalibrationsDefinition:
    """
    A class to manage schedules with calibrated parameter values.
    Schedules are stored in a dict and are intended to be fully parameterized,
    including the index of the channels. The parameters are therefore stored in
    the schedules. The names of the parameters must be unique. The calibrated
    values of the parameters are stored in a dictionary.
    """

    def __init__(self, backend):
        """
        Initialize the instructions from a given backend.

        Args:
            backend: The backend from which to get the configuration.
        """

        self._n_qubits = backend.configuration().num_qubits
        self._n_uchannels = backend.configuration().n_uchannels
        self._properties = backend.properties()
        self._config = backend.configuration()
        self._params = {'qubit_freq': {}}
        self._schedules = {}

        # Populate the qubit frequency estimates
        for qubit, freq in enumerate(backend.defaults().qubit_freq_est):
            timestamp = backend.properties().qubit_property(qubit)['frequency'][1]
            val = ParameterValue(freq, timestamp)
            self.add_parameter_value('qubit_freq', val, DriveChannel(qubit))

    def schedules(self) -> pd.DataFrame:
        """
        Return the schedules in self in a data frame to help
        users manage their schedules.

        Returns:
            data: A pandas data frame with all the schedules in it.
        """
        data = []
        for name, schedule in self._schedules.items():
            data.append({'name': name,
                         'schedule': schedule,
                         'parameters': schedule.parameters})

        return pd.DataFrame(data)

    def parameters(self, names: Optional[List[str]] = None,
                   chs: Optional[List[PulseChannel]] = None) -> pd.DataFrame:
        """
        Returns the parameters as a pandas data frame.
        This function is here to help users manage their parameters.

        Args:
            names: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            chs: The channels that should be included in the returned table.
                If None is given then all channels are returned.

        Returns:
            data: A data frame of parameter values.
        """

        data = []

        if names is None:
            names = self._params.keys()

        for name in names:
            params_name = self._params.get(name, {})

            if chs is None:
                chs = params_name.keys()

            for ch in chs:
                for value in params_name.get(ch, {}):
                    value_dict = dataclasses.asdict(value)
                    value_dict['channel'] = ch.name
                    value_dict['parameter'] = name

                    data.append(value_dict)

        return pd.DataFrame(data)

    def add_schedules(self, schedules: Union[Schedule, List[Schedule]]):
        """
        Add a schedule and register the parameters.

        Args:
            schedules: The schedule to add.

        Raises:
            CalibrationError: If the parameterized channel index is not formatted
                following index1.index2...
        """
        if isinstance(schedules, Schedule):
            schedules = [schedules]

        for schedule in schedules:

            # check that channels, if parameterized, have the proper name format.
            #pylint: disable = raise-missing-from
            for ch in schedule.channels:
                if isinstance(ch.index, Parameter):
                    try:
                        [int(index) for index in ch.index.name.split('.')]
                    except ValueError:
                        raise CalibrationError('Parameterized channel must have a name '
                                               'formatted following index1.index2...')

            self._schedules[schedule.name] = schedule

            for param in schedule.parameters:
                if param.name not in self._params:
                    self._params[param.name] = {}

    def add_parameter_value(self, param: Union[Parameter, str],
                            value: ParameterValue,
                            chs: Optional[Union[PulseChannel, List[PulseChannel]]] = None,
                            ch_type: Type[PulseChannel] = None):
        """
        Add a parameter value to the stored parameters. This parameter value may be
        applied to several channels, for instance, all DRAG pulses may have the same
        standard deviation. The parameters are stored and identified by name.

        Args:
            param: The parameter or its name for which to add the measured value.
            value: The value of the parameter to add.
            chs: The channel(s) to which the parameter applies. If None is given
                then the type of channels must by specified.
            ch_type: This parameter is only used if chs is None. In this case the
                value of the parameter will be set for all channels of the
                specified type.

        Raises:
            CalibrationError: if ch_type is not given when chs are None, if the
                channel type is not a ControlChannel, DriveChannel, or MeasureChannel, or
                if the parameter name is not already in self.
        """
        if isinstance(param, Parameter):
            name = param.name
        else:
            name = param

        if chs is None:
            if ch_type is None:
                raise CalibrationError('Channel type must be given when chs are None.')

            if issubclass(ch_type, ControlChannel):
                chs = [ch_type(_) for _ in range(self._n_uchannels)]
            elif issubclass(ch_type, (DriveChannel, MeasureChannel)):
                chs = [ch_type(_) for _ in range(self._n_qubits)]
            else:
                raise CalibrationError('Unrecognised channel type {}.'.format(ch_type))

        try:
            chs = list(chs)
        except TypeError:
            chs = [chs]

        if name not in self._params:
            raise CalibrationError('Cannot add unknown parameter %s.' % name)

        for ch in chs:
            if ch not in self._params[name]:
                self._params[name][ch] = [value]
            else:
                self._params[name][ch].append(value)

    def get_channel_index(self, qubits: Tuple, chan: PulseChannel) -> int:
        """
        Get the index of the parameterized channel based on the given qubits
        and the name of the parameter in the channel index. The name of this
        parameter must be written as qubit_index1.qubit_index2... . For example,
        the following parameter names are valid: '1', '1.0', '3.10.0'.

        Args:
            qubits: The qubits for which we want to obtain the channel index.
            chan: The channel with a parameterized name.

        Returns:
            index: The index of the channel. For example, if qubits=(int, int) and
                the channel is a u channel with parameterized index name 'x.y'
                where x and y the method returns the u_channel corresponding to
                qubits (qubits[1], qubits[0]).

        Raises:
            CalibrationError: if the number of qubits is incorrect, if the
                number of inferred ControlChannels is not correct, or if ch is not
                a DriveChannel, MeasureChannel, or ControlChannel.
        """

        if isinstance(chan.index, Parameter):
            indices = [int(_) for _ in chan.index.name.split('.')]
            ch_qubits = tuple(qubits[_] for _ in indices)

            if isinstance(chan, DriveChannel):
                if len(ch_qubits) != 1:
                    raise CalibrationError('Too many qubits for drive channel: '
                                      'got %i expecting 1.' % len(ch_qubits))

                ch_ = self._config.drive(ch_qubits[0])

            elif isinstance(chan, MeasureChannel):
                if len(ch_qubits) != 1:
                    raise CalibrationError('Too many qubits for drive channel: '
                                      'got %i expecting 1.' % len(ch_qubits))

                ch_ = self._config.measure(ch_qubits[0])

            elif isinstance(chan, ControlChannel):
                chs_ = self._config.control(ch_qubits)

                if len(chs_) != 1:
                    raise CalibrationError('Ambiguous number of control channels for '
                                      'qubits {} and {}.'.format(qubits, chan.name))

                ch_ = chs_[0]

            else:
                chs = tuple(_.__name__ for _ in [DriveChannel, ControlChannel, MeasureChannel])
                raise CalibrationError('Channel must be of type {}.'.format(chs))

            return ch_.index
        else:
            return chan.index

    def parameter_value(self, name: str, chan: PulseChannel, valid_only: bool = True,
                        group: str = 'default',
                        cutoff_date: datetime = None) -> Union[int, float, complex]:
        """
        Retrieve the value of a calibrated parameter from those stored.

        Args:
            name: The name of the parameter to get.
            chan: The channel for which we want the value of the parameter.
            valid_only: Use only parameters marked as valid.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date.
                Parameters generated after the cutoff date will be ignored. If the
                cutoff_date is None then all parameters are considered.

        Returns:
            value: The value of the parameter.

        Raises:
            CalibrationError: if there is no parameter value for the given parameter name
                and pulse channel.
        """
        #pylint: disable = raise-missing-from
        try:
            if valid_only:
                candidates = [p for p in self._params[name][chan] if p.valid]
            else:
                candidates = self._params[name][chan]

            candidates = [_ for _ in candidates if _.group == group]

            if cutoff_date:
                candidates = [_ for _ in candidates if _ <= cutoff_date]

            candidates.sort(key=lambda x: x.date_time)

            return candidates[-1].value
        except KeyError:
            raise CalibrationError('No parameter value for %s and channel %s' % (name, chan.name))

    def get_schedule(self, name: str, qubits: Tuple[int, ...],
                     free_params: List[str] = None, group: Optional[str] = 'default') -> Schedule:
        """
        Args:
            name: The name of the schedule to get.
            qubits: The qubits for which to get the schedule.
            free_params: The parameters that should remain unassigned.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.

        Returns:
            schedule: A deep copy of the template schedule with all parameters assigned
                except for those specified by free_params.

        Raises:
            CalibrationError: if the name of the schedule is not known.
        """

        # Get the schedule and deepcopy it to prevent binding from removing
        # the parametric nature of the schedule.
        if name not in self._schedules:
            raise CalibrationError('Schedule %s is not defined.' % name)

        sched = copy.deepcopy(self._schedules[name])

        # Retrieve the channel indices based on the qubits and bind them.
        binding_dict = {}
        for ch in sched.channels:
            if ch.is_parameterized():
                index = self.get_channel_index(qubits, ch)
                binding_dict[ch.index] = index

        sched.assign_parameters(binding_dict)

        # Loop through the remaining parameters in the schedule, get their values and bind.
        if free_params is None:
            free_params = []

        binding_dict = {}
        for inst in sched.instructions:
            ch = inst[1].channel
            for param in inst[1].parameters:
                if param.name not in free_params:
                    binding_dict[param] = self.parameter_value(param.name, ch, group=group)

        sched.assign_parameters(binding_dict)

        return sched

    def get_circuit(self, name: str, qubits: Tuple, free_params: List[str] = None,
                    group: Optional[str] = 'default', schedule: Schedule = None) -> QuantumCircuit:
        """
        Args:
            name: The name of the gate to retrieve.
            qubits: The qubits for which to generate the Gate.
            free_params: Names of the parameters that will remain unassigned.
            group: The calibration group from which to retrieve the calibrated values.
                If unspecified this default to 'default'.
            schedule: The schedule to add to the gate if the internally stored one is
                not going to be used.

        Returns:
            A quantum circuit in which the parameter values have been assigned aside from
            those explicitly specified in free_params.
        """
        if schedule is None:
            schedule = self.get_schedule(name, qubits, free_params, group)

        gate = Gate(name=name, num_qubits=len(qubits), params=list(schedule.parameters))
        circ = QuantumCircuit(len(qubits), len(qubits))
        circ.append(gate, list(range(len(qubits))))
        circ.add_calibration(gate, qubits, schedule, params=schedule.parameters)

        return circ

    def to_db(self):
        """
        Serializes the parameterized schedules and parameter values so
        that they can be sent and stored in an external DB.
        """
        raise NotImplementedError

    def from_db(self):
        """
        Retrieves the parameterized schedules and pulse parameters from an
        external DB.
        """
        raise NotImplementedError
