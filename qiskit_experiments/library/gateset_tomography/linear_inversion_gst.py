import numpy as np
from typing import Dict
from qiskit.quantum_info import PTM, Choi


def linear_inversion_gst(outcome_data, gateset_basis) -> Dict[str, PTM]:

    """
    Reconstruct a gate set from measurement data using linear inversion.

    Returns:l
        For each gate in the gateset: its approximation found
        using the linear inversion process.

    Additional Information:
        Given a gate set (G1,...,Gm)
        and SPAM circuits (F1,...,Fn) constructed from those gates
        the data should contain the probabilities of the following types:
        p_ijk = E*F_i*G_k*F_j*rho
        p_ij = E*F_i*F_j*rho

        We have p_ijk = self.probs[(Fj, Gk, Fi)] since in self.probs
        (Fj, Gk, Fi) indicates first applying Fj, then Gk, then Fi.

        One constructs the Gram matrix g = (p_ij)_ij
        which can be described as a product g=AB
        where A = sum (i> <E F_i) and B=sum (F_j rho><j)
        For each gate Gk one can also construct the matrix Mk=(pijk)_ij
        which can be described as Mk=A*Gk*B
        Inverting g we obtain g^-1 = B^-1A^-1 and so
        g^1 * Mk = B^-1 * Gk * B
        This gives us a matrix similiar to Gk's representing matrix.
        However, it will not be the same as Gk,
        since the observable results cannot distinguish
        between (G1,...,Gm) and (B^-1*G1*B,...,B^-1*Gm*B)
        a further step of *Gauge optimization* is required on the results
        of the linear inversion stage.
        One can also use the linear inversion results as a starting point
        for a MLE optimization for finding a physical gateset, since
        unless the probabilities are accurate, the resulting gateset
        need not be physical.
    """
    n = len(gateset_basis.spam_labels)
    m = len(gateset_basis.gate_labels)
    gram_matrix = np.zeros((n, n))
    E = np.zeros((1, n))
    rho = np.zeros((n, 1))
    gate_matrices = []
    for i in range(m):
        gate_matrices.append(np.zeros((n, n)))

    for i in range(n):  # row
        F_i = gateset_basis.spam_labels[i]
        E[0][i] = outcome_data[(F_i,)]
        rho[i][0] = outcome_data[(F_i,)]
        for j in range(n):  # column
            F_j = gateset_basis.spam_labels[j]
            gram_matrix[i][j] = outcome_data[(F_i, F_j)]

            for k in range(m):  # gate
                G_k = gateset_basis.gate_labels[k]
                gate_matrices[k][i][j] = outcome_data[(F_i, G_k, F_j)]

    gram_inverse = np.linalg.inv(gram_matrix)

    gates = [PTM(gram_inverse @ gate_matrix) for gate_matrix in gate_matrices]
    result = dict(zip(gateset_basis.gate_labels, gates))
    result["E"] = E
    result["rho"] = gram_inverse @ rho
    return result
