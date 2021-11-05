from qiskit.extensions import XGate, YGate, HGate, U2Gate

from gst_experiment import GateSetTomography
from qiskit import Aer
import numpy as np

backend = Aer.get_backend('qasm_simulator')

basis_1qubit= {
 'Id': lambda circ, qubit: None,
 'H': lambda circ, qubit: circ.append(HGate(), [qubit]),
 'Y': lambda circ, qubit: circ.append(YGate(), [qubit]),
 'X_Rot_90': lambda circ, qubit: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit]),
}
basis_2qubits = {
 'IdId': lambda circ, qubit1, qubit2: None,
 'X I': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit2]),
 'Y I': lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit2]),
 'I X': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit1]),
 'I Y': lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit1]),
 'H I': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit2]),
 'I H': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit1]),
 'I X_Rot_90': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit1]),
 'X_Rot_90 I': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit2]),
  # qubit1 is the ctrl qubit, qubit2 is the target
}

#test #1:
"""
qstexp1 = GateSetTomography(qubits=[0], gateset=basis_1qubit,additional_gates=[HGate()], only_basis_gates=True)
qstexp1.set_analysis_options(fitter='scipy_optimizer_MLE_gst', fitter_initial_guess='linear_inversion')
qstdata1 = qstexp1.run(backend).block_for_results()
print(qstdata1.analysis_results("gst gateset results"))
print(qstdata1.analysis_results("gst fidelity with ideal target"))
print(qstdata1.analysis_results("Hilbert-Schmidtt norm for the target and gst results"))
print(qstdata1.analysis_results("froebenius distance between target and gst results"))
"""

"""

#test #2:

qstexp1 = GateSetTomography(qubits=[0], gateset='default',additional_gates=[HGate()])
qstexp1.set_analysis_options(fitter='scipy_optimizer_MLE_gst', fitter_initial_guess='linear_inversion')
qstdata1 = qstexp1.run(backend).block_for_results()
print(qstdata1.analysis_results("gst gateset results"))
print('extra:\n', qstdata1.analysis_results("gst gateset results").extra)
print(qstdata1.analysis_results("gst fidelity with ideal target"))
print(qstdata1.analysis_results("Hilbert-Schmidtt norm for the target and gst results"))
print(qstdata1.analysis_results("froebenius distance between target and gst results"))
"""

"""
qstexp1 = GateSetTomography(qubits=2,measured_qubits=[0,1], gateset='default')
qstexp1.set_analysis_options(fitter='scipy_optimizer_MLE_gst', fitter_initial_guess='linear_inversion')
qstdata1 = qstexp1.run(backend).block_for_results()
print(qstdata1.analysis_results("gst gateset results"))
print('extra:\n', qstdata1.analysis_results("gst gateset results").extra)
print(qstdata1.analysis_results("gst fidelity with ideal target"))
print(qstdata1.analysis_results("Hilbert-Schmidtt norm for the target and gst results"))
print(qstdata1.analysis_results("froebenius distance between target and gst results"))
"""

qstexp1 = GateSetTomography(qubits=[0], gateset='default')
qstexp1.set_analysis_options(fitter='scipy_optimizer_MLE_gst', fitter_initial_guess='linear_inversion')
qstdata1 = qstexp1.run(backend).block_for_results()
print(qstdata1.analysis_results("GST Experiment properties"))
print(qstdata1.analysis_results("gst estimation of Id"))
print('extra:\n', qstdata1.analysis_results("gst estimation of Id").extra)
