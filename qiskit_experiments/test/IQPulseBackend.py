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
            for circuit in run_inputs:
                if len(circuit.calibrations) == 0:
                    schedule = schedule(circuit, self)# self position is for the backend : is self right?
                elif circuit.calibrations : 
                    #circuit이 cal가지고있으면 
                    #cal을 뽑아낸다음 addcal한 최종 circuit을 스케쥴로 만든다
                    # cal 가진 circuit은 자동으로 cal 반영해서 생성되는거같고
                    # schedule -> solver.solve
                    circuit_unitraires = {}
                    circuit.add_calibration()
                    for inst_name, instructions in circuit.calibrations.items():
                        circuit_unitraires[inst_name] = instructions
                        for qubits_params, schedule in instructions.items():




            return FakeJob(self, Result.from_dict(result))