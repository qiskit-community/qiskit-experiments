import numpy as np
from qiskit import circuit
from qiskit import QuantumCircuit
from qiskit import pulse
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.states import Statevector
from qiskit.providers import BackendV2
from qiskit.circuit.library.standard_gates import XGate, SXGate, IGate

from qiskit.providers.aer import PulseSimulator
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.compiler import assemble
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.library import calibration
from qiskit_experiments.test.utils import FakeJob
from qiskit.result import Result

from qiskit_dynamics import Solver, Signal
from qiskit_dynamics.pulse import InstructionToSignals

class IQPulseBackend(BackendV2):

    def __init__(self, hamiltonian_static, hamiltonian_operator,  dt: float = 0.222):
        '''Hamiltonian and operators is the Qiskit Dynamics object'''
        self.hamiltonian_static=hamiltonian_static
        self.hamiltonian_operator=hamiltonian_operator
        self.dt=dt
        self._solver = Solver(self.hamiltonian_static, self.hamiltonian_operator)  

    def _state_vector_to_result(self, state: Statevector, meas_return: MeasReturnType, meas_level: MeasLevel):
    """Convert the state vector to IQ data or counts."""
    
    if meas_level == MeasLevel.CLASSIFIED:
        counts = ... # sample from the state vector. There might already be functions in Qiskit to do this.
    if meas_level == MeasLevel.KERNELED:
        iq_data = ... create IQ data.

    def run(self, run_inputs, meas_level, shots, meas_return) -> FakeJob:
        ''' This should be able to return both counts and IQ data depending on the run options.
        Loop through each circuit, extract the pulses from the circuit.calibraitons
        and give them to the pulse solver.
        Multiply the unitaries of the gates together to get a state vector
        that we convert to counts or IQ Data'''
        run_inputs= self.run_inputs        
        
        if isinstance(run_inputs, QuantumCircuit):
            run_inputs = [run_inputs]
        # extract calibration->get the pulse schedule->
        # unitary = simulated pulse schedule 
        circuit_unitaries = {}
        for circuit in run_inputs:
            for inst_name, instructions in circuit.calibrations.items():
                circuit_unitaries[inst_name] = instructions
                for qubit_params, schedule in instructions.items():
                    solver = Solver(
                    static_hamiltonian = self.hamiltonian_static,
                    hamiltonian_operators = self.hamiltonian_operator,
                    rotating_frame = self.hamiltonian_static,
                    rwa_cutoff_freq=2*5.0,
                    hamiltonian_channels = ['d0'] # how to define what qubit to use?
                    channel_carrier_freqs = {'d0' : w}, #how to define what qubit to use?
                    dt = 0.222
                    )
                # gave a specific number just for convenience for the time
                    sol = solver.solve(t_span=[0., 20.], y0=np.eye(self._dim), 
                                        signals = schedule, t_eval=np.linspace(0., 200, 2000))
                # how to extract unitary from solver.solve
                    circuit_unitaries[inst_name][qubit_params] = unitary # suppose we extacted unitary from solver.solve
        # multiply the unitaries for the circuit(only for single qubit)
            calibration_unitary=[]
            for index,(key,elem) in enumerate(circuit.calibrations.items()):
                calibration_unitary.append(key)
              
            total_circuit_unitary=[] # cal 안된 애는 cir.data거 그대로 쓰면되고 cal 된애는 unitary above써야되
            for i in range(len(circuit.data)):
                unitary = circuit.data[i][0].name
                if unitary in calibration_unitary:
                    unitary = #given by solver.solve which is in circuit_unitaries
                elif unitary not in calibration_unitary:
                    if unitary == 'x':
                        unitary = Operator(XGate())
                    elif unitary == 'sx':
                        unitary = Operator(SXGate())
                total_circuit_unitary.append(unitary)
            for unitary in total_circuit_unitary:
                single_unitary = Operator(IGate()).compose(unitary)
        # solver.solve(single_unitary)   
        # psi = np.array([[1], [0], [0]])  # Need to be a bit more clever here with the dimensions
        # psi = unitary @ psi  # Forward unitary dynamics.
        #result = self._statevector_to_result(psi, meas_return, meas_level)

        return FakeJob(self, Result.from_dict(result))
       