from qiskit import circuit
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

    def run(self, run_input, meas_level, shots, meas_return) -> FakeJob:
        ''' This should be able to return both counts and IQ data depending on the run options.
         Loop through each circuit, extract the pulses from the circuit.calibraitons
          and give them to the pulse solver.
          Multiply the unitaries of the gates together to get a state vector that we convert to counts or IQ Data'''
        run_input= self.run_input
        # run_input a list of circuits
        pulse_schedule = []
        pulse_job = None
        
        if isinstance(run_input, (pulse.Schedule, pulse,ScheduleBlock)):
            pulse_job = True
        elif isinstance(run_input, circuit.QuantumCircuit):
            #extracting pulse from circuit : need help!
        elif isinstacne(run_input, FineXAMplitude):# for all other calibration experiments
            pulse_job = True
            exp_circuits=run_input.circuits()
            for circuit in exp_circuits:
                #extract pulse from each circuit
                pulse_schedule.append(#pulse)
        if pulse_job:
            converter = InstructionToSignals(self.dt, carrier={"d0" : w})#need to define carrier freq
            signals = converter.get_signal(pulse_schedule[0])
            hamiltonian_solver = Solver(
                static_hamiltonian = self.hamiltonian_static,
                hamiltonina_operator = self.hamiltonian_operator,
                rotating_frame = self.hamiltonian_static,
                hamiltonian_channels = ['d0'] #define which channel to use
                channel_carrier_freqs = {'d0', : w},#define w
                dt = self.dt
            )
            initial_state = Statevector([1., 0.])
            # t_span, signals need fix
            sol = hamiltonian_solver.solve(t_span=[0., 30.], y0 = initial_state,
                                         signals=signals, atol=1e-8, rtol=1e-8)
            
            #need to consider how to process IQ

            return FakeJob(self, Result.from_dict(result))