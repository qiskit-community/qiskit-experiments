import numpy as np
from qiskit import circuit
from qiskit import QuantumCircuit
from qiskit import pulse
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.states import Statevector

from qiskit.providers import BackendV2

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

    def run(self, run_inputs, meas_level, shots, meas_return) -> FakeJob:
        ''' This should be able to return both counts and IQ data depending on the run options.
         Loop through each circuit, extract the pulses from the circuit.calibraitons
          and give them to the pulse solver.
          Multiply the unitaries of the gates together to get a state vector that we convert to counts or IQ Data'''
        run_inputs= self.run_inputs        
        
        if isinstance(run_inputs, QuantumCircuit):
            run_inputs = [run_inputs]
            schedules = []
            for circuit in run_inputs:
                # divide the case in two 
                # when circuit has it's own calibration or no calibration
                if len(circuit.calibrations) == 0: # No cal -> extract the schedule directly
                    schedule = schedule(circuit, self)# self position is for the backend : is self right? 
                    # How can I give the backend as an argument??
                    schedules.append(schedule)
                elif circuit.calibrations : # eventhough there's cal, directly extracted schedule include that cal
                    schedule = schedule(circuit, self)
                    schedules.append(schedule)
                # simulate the circuit's extracted schedult with the solver
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
                sol = solver.solve(t_span=[0., 20.], y0=Statevector([1., 0.]), 
                                    signals = schedules, t_eval=np.linspace(0., 200, 2000))
###################################################################################
# This is the most confusing part. I dont understand the necessity of this part.
# Isnt schedule(circuit, backend) enough for extracting the pulse shcedule??
# Why calibration.items?? I checked If the experiment include calibration, the generated circuit reflect the calibration already.              
        circuit_unitaries = {}
        circuit.add_calibration()
        for inst_name, instructions in circuit.calibrations.items():
            circuit_unitraires[inst_name] = instructions
            for qubits_params, schedule in instructions.items():       
                
                unitary = self._run_pulse_simulation(schedule)#Does it mean Solver.solve?? or the real pulse simulator??
                 
                circuit_unitaries[inst_name][qubits_params] = unitary
         # multiply the unitaries for the circuit
        psi = np.array([[1], [0], [0]])  # Need to be a bit more clever here with the dimensions
        for inst in circuit.data:  # TODO check
            qubits, params, inst_name = self._get_info(inst)  # TODO
                
            unitary = circuit_unitaries[inst_name][(qubits, params)]
                
            psi = unitary @ psi  # Forward unitary dynamics.




            return FakeJob(self, Result.from_dict(result))