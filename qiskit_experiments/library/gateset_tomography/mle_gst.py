import scipy.optimize as opt
import numpy as np
import itertools
from qiskit.quantum_info import PTM, Choi, Operator, DensityMatrix
from typing import Tuple, List, Optional, Dict
from gauge_optimizer import Pauli_strings
import scipy.linalg as la
from functools import reduce
from qiskit_experiments.library.tomography.fitters.fitter_utils import make_positive_semidefinite

class GST_Optimize:
    """ GST fitter that performs the maximum likelihood estimation (MLE) optimization for gate set tomography.

    # section: overview
        Maximum liklihood estimation method aims to find the predicted true probabilities :math:`p_{ijk}` based on the GST
        experiment measurements `m_{ijk}` from which the most-likely gate set that produced the
        experimental data is found. The main advantage of this optimization method, is that it does so
        together with making sure the physical constraints are satisfied unlike the linear inversion solution.

        This experiment protocol is based on the protocol described in section 3.5.2 in
        arXiv:1509.02921 in which Pauli transfer matrix representation is used as the gates representation
        to solve the MLE optimization problem.

    # section: reference
        .. ref_arxiv:: 1 1509.02921
    """

    def __init__(self,
                 Gs: List[str],
                 Fs_names: Tuple[str],
                 Fs: Dict[str, Tuple[str]],
                 probs: Dict[Tuple[str], float],
                 initial_gateset: Dict[str, PTM],
                 num_qubits: int,
                 ):
        """Initializes the data for the MLE optimizer
        Args:
            Gs: The names of the gates in the gateset
            Fs_names: The names of the SPAM circuits
            Fs: The SPAM specification (SPAM name -> gate names)
            probs: The probabilities obtained experimentally :math:`{m_ijk}`
            initial_gateset: The gate set that is used as an initial guess. As default,
            the linear inversion gate set solution is used. It can also be None, or
            an arbitrary initial guess provided by the user.
            num_qubits: number of qubits
        """

        self.probs = probs
        self.Gs = Gs
        self.Fs_names = Fs_names
        self.Fs = Fs
        self.num_qubits = num_qubits
        self.obj_fn_data = self._compute_objective_function_data()

        if initial_gateset is not None:
            self.initial_value = self.set_initial_value(initial_gateset)
        else:
            self.initial_value = None


    # auxiliary functions
    @staticmethod
    def _split_list(input_list: List, sizes: List) -> List[List]:
        """Splits a list to several lists of given size
        Args:
            input_list: A list
            sizes: The sizes of the splitted lists
        Returns:
            list: The splitted lists
        Example:
            >> split_list([1,2,3,4,5,6,7], [1,4,2])
            [[1],[2,3,4,5],[6,7]]

        Raises:
            QiskitError: if length of l does not equal sum of sizes
        """
        if sum(sizes) != len(input_list):
            msg = "Length of list ({}) " \
                  "differs from sum of split sizes ({})".format(len(input_list), sizes)
            raise QiskitError(msg)
        result = []
        i = 0
        for s in sizes:
            result.append(input_list[i:i + s])
            i = i + s
        return result

    @staticmethod
    def _vec_to_complex_matrix(vec: np.array) -> np.array:
        n = int(np.sqrt(vec.size / 2))
        if 2 * n * n != vec.size:
            raise QiskitError("Vector of length {} cannot be reshaped"
                               " to square matrix".format(vec.size))
        size = n * n
        return np.reshape(vec[0:size] + 1j * vec[size: 2 * size], (n, n))

    @staticmethod
    def _complex_matrix_to_vec(M):
        mvec = M.reshape(M.size)
        return list(np.concatenate([mvec.real, mvec.imag]))

    def _compute_objective_function_data(self) -> List:
        """Computes auxiliary data needed for efficient computation
        of the objective function.

        Returns:
             The objective function data list
        Additional information:
            The objective function is
            sum_{ijk}(<|E*R_Fi*G_k*R_Fj*Rho|>-m_{ijk})^2
            We expand R_Fi*G_k*R_Fj to a sequence of G-gates and store
            indices. We also obtain the m_{ijk} value from the probs list
            all that remains when computing the function is thus
            performing the matrix multiplications and remaining algebra.
        """
        m = len(self.Fs)
        n = len(self.Gs)
        obj_fn_data = []
        for (i, j) in itertools.product(range(m), repeat=2):
            for k in range(n):
                Fi = self.Fs_names[i]
                Fj = self.Fs_names[j]
                m_ijk = (self.probs[(Fj, self.Gs[k], Fi)])
                Fi_matrices = [self.Gs.index(gate) for gate in self.Fs[Fi]]
                Fj_matrices = [self.Gs.index(gate) for gate in self.Fs[Fj]]
                matrices = Fj_matrices + [k] + Fi_matrices
                obj_fn_data.append((matrices, m_ijk))
        return obj_fn_data

    def _split_input_vector(self, x: np.array, fullness: str) -> Tuple:
        """Reconstruct the GST data from its vector representation
        Args:
            x: The vector representation of the GST data
            fullness: Takes two values, 'full': in case x has no missing elements and '0': in case the
            last elements of each corresponding lower triangle matrix of the Cholesky decomposition of
            the gates are missing.

        Returns:
            The GST data (E, rho, Gs) (see additional info)

        Additional information:
            The gate set tomography data is a tuple (E, rho, Gs) consisting of
            1) A POVM measurement operator E
            2) An initial quantum state rho
            3) A list Gs = (G1, G2, ..., Gk) of gates, represented as matrices

            This function reconstructs (E, rho, Gs) from the vector x
            Since the MLE optimization procedure has PSD constraints on
            E, rho and the Choi represetnation of the PTM of the Gs,
            we rely on the following property: M is PSD iff there exists
            T such that M = T @ T^{dagger}.
            Hence, x stores those T matrices for E, rho and the Gs

            if x has missing values (for the sake of enforcing TP constraint),
            it replaces them with zeros and returns the corresponding GST data.
        """
        n = len(self.Gs)
        d = (2 ** self.num_qubits)
        ds = d ** 2  # d squared - the dimension of the density operator

        d_t = 2 * d ** 2
        ds_t = 2 * ds ** 2

        if fullness == 'full':
            T_vars = self._split_list(x, [d_t, d_t] + [ds_t] * n)
        else:  # =='0'
            T_vars = self._split_list(x, [d_t, d_t] + [ds_t - 1] * n)
            # add zeros instead of last real element and it will be changed in the obj func
            for i in range(n):
                T_vars[2 + i] = np.insert(T_vars[2 + i], ds ** 2 - 1, 0)

        E_T = self._vec_to_complex_matrix(T_vars[0])
        rho_T = self._vec_to_complex_matrix(T_vars[1])

        Gs_T = [self._vec_to_complex_matrix(T_vars[2 + i]) for i in range(n)]

        E = np.reshape(E_T @ np.conj(E_T.T), (1, ds))
        rho = np.reshape(rho_T @ np.conj(rho_T.T), (ds, 1))
        Gs = [PTM(Choi(G_T @ np.conj(G_T.T))).data for G_T in Gs_T]

        return (E, rho, Gs)

    def _join_input_vector(self,
                           E: np.array,
                           rho: np.array,
                           Gs: List[np.array]
                           ) -> np.array:
        """Converts the GST data into a vector representation
        Args:
            E: The POVM measurement operator
            rho: The initial state
            Gs: The gates list

        Returns:
            The vector representation of (E, rho, Gs)

        Additional information:
            The GST data is encoded in terms of Cholesky decomposition to make sure: E, rho and every
             G in the gateset are all PSD.
            This function performs the inverse operation to
            split_input_vector; the notations are the same.
        """
        d = (2 ** self.num_qubits)
        #np.linalg returns the lower diagonal matrix T of the Cholesky decomposition T_dagger*T, but it
        #works only with positive definite matrix. To solve this, we simply add a small matrix to shift
        # the zero eigenvalues.

        E_T = np.linalg.cholesky(make_positive_semidefinite(E.reshape((d, d))) + 1e-14 * np.eye(d))
        rho_T = np.linalg.cholesky(make_positive_semidefinite(rho.reshape((d, d))) + 1e-14 * np.eye(d))

        Gs_Choi = [Choi(PTM(G)).data for G in Gs]
        Gs_Choi_mod = [make_positive_semidefinite(G) + 1e-14 * np.eye(d ** 2) for G in Gs_Choi]
        Gs_T = [np.linalg.cholesky(G) for G in Gs_Choi_mod]
        E_vec = self._complex_matrix_to_vec(E_T)
        rho_vec = self._complex_matrix_to_vec(rho_T)
        result = E_vec + rho_vec
        for G_T in Gs_T:
            result += self._complex_matrix_to_vec(G_T)
        return np.array(result)

    def _obj_fn(self, x: np.array) -> float:
        """The MLE objective function
        Args:
            x: The vector representation of the GST data (E, rho, Gs) with the last elements of the
            Cholesky matrices corresponding to all the gates in the gateset are missing

        Returns:
            The MLE cost function (see additional information)

        Additional information:
            The MLE objective function is obtained by approximating
            the MLE estimator using the central limit theorem.

            It is computed as the sum of all terms of the form
            (m_{ijk} - p_{ijk})^2
            Where m_{ijk} are the experimental results, and
            p_{ijk} are the predicted results for the given GST data:
            p_{ijk} = E*F_i*G_k*F_j*rho.

            For additional info, see section 3.5 in arXiv:1509.02921
            and about the filling of the x vector, see the information in _complete_x method below.
        """
        E, rho, G_matrices = self._split_input_vector(x, '0')
        n = len(G_matrices)
        x1 = np.copy(x)
        x2 = np.copy(x)
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        l = 0
        for i in range(n):
            m = 0
            for j in range(2 * ds ** 2 - 1):
                m += (abs(x[4 * ds + 2 * i * ds ** 2 - l + j])) ** 2
            l += 1
            x1 = np.insert(x1, 4 * ds + (2 * i + 1) * ds ** 2, np.sqrt(abs(d - m)))
            x2 = np.insert(x2, 4 * ds + (2 * i + 1) * ds ** 2, -np.sqrt(abs(d - m)))

        # E, rho, G_matrices = self._split_input_vector(x)
        E, rho, G_matrices = self._split_input_vector(x1, 'full')
        val1 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val1 = val1 + term_val

        E, rho, G_matrices = self._split_input_vector(x2, 'full')
        val2 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val2 = val2 + term_val
        return np.min([val1, val2])

    def _complete_x(self, x: np.array) -> float:
        """ Completes the x vector by adding the suitable elements to fill the deleted elements; last
        element of each Cholesky decomposition matrices _T of the gates. The suitable elements are
        those that give trace=2**num_qubits for the Choi matrix corresponding to each gate in the gateset.
        As the trace does not depend on the signs of these elements, their sign is picked as the one that
        returns lower estimation of the objection function.

        Args:
            x: The vector representation of the GST data (E, rho, Gs) with the last elements of the
            Cholesky matrices corresponding to all the gates in the gateset are missing

        Returns:
            The full x vector representation of the GST data
        """
        E, rho, G_matrices = self._split_input_vector(x, '0')
        n = len(G_matrices)
        x1 = np.copy(x)
        x2 = np.copy(x)
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        l = 0
        for i in range(n):
            m = 0

            for j in range(2 * ds ** 2 - 1):
                m += (abs(x[4 * ds + 2 * i * ds ** 2 - l + j])) ** 2
            l += 1
            x1 = np.insert(x1, 4 * ds + (2 * i + 1) * ds ** 2, np.sqrt(abs(d - m)))
            x2 = np.insert(x2, 4 * ds + (2 * i + 1) * ds ** 2, -np.sqrt(abs(d - m)))

        E, rho, G_matrices = self._split_input_vector(x1, 'full')
        val1 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val1 = val1 + term_val

        E, rho, G_matrices = self._split_input_vector(x2, 'full')
        val2 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val2 = val2 + term_val

        if (np.min([val1, val2]) == val1):
            x = x1
        else:
            x = x2
        return x

    def _ptm_matrix_values(self, x: np.array) -> List[np.array]:
        """Returns a vectorization of the gates matrices
        Args:
            x: The vector representation of the GST data

        Returns:
            A vectorization of all the PTM matrices for the gates
            in the GST data

        Additional information:
            This function is not trivial since the returned vector
            is not a subset of x, since for each gate G, what x
            stores in practice is a matrix T, such that the
            Choi matrix of G is T@T^{dagger}. This needs to be
            converted into the PTM representation of G.
        """
        xfull = self._complete_x(x)
        _, _, G_matrices = self._split_input_vector(xfull, 'full')
        result = []
        for G in G_matrices:
            result = result + self._complex_matrix_to_vec(G)
        return result

    def _rho_trace(self, x: np.array) -> Tuple[float]:
        """Returns the trace of the GST initial state
        Args:
            x: The vector representation of the GST data
        Returns:
            The trace of rho - the initial state of the GST. The real
            and imaginary part are returned separately.
        """
        xfull = self._complete_x(x)
        _, rho, _ = self._split_input_vector(xfull, 'full')
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        rho = self._convert_from_ptm(rho.reshape((d, d)))
        trace = sum([rho[i][i] for i in range(d)])
        return (np.real(trace), np.imag(trace))

    def _bounds_eq_constraint(self, x: np.array) -> List[float]:
        """Equality MLE constraints on the GST data

        Args:
            x: The vector representation of the GST data

        Returns:
            The list of computed constraint values (should equal 0)

        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) G_{0,0} is 1 for every gate G
            2) The rest of the first row of each G is 0.
            3) G only has real values, so imaginary part is 0.

            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self._ptm_matrix_values(x)
        bounds_eq = []
        n = len(self.Gs)
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            bounds_eq.append(ptm_matrix[i] - 1)  # G^k_{0,0} is 1
            i += 1
            for _ in range(ds - 1):
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
        return bounds_eq

    def _bounds_ineq_constraint(self, x: np.array) -> List[float]:
        """Inequality MLE constraints on the GST data

        Args:
            x: The vector representation of the GST data

        Returns:
            The list of computed constraint values (should be >= 0)

        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) Every row of G except the first has entries in [-1,1]

            We implement this as two inequalities per entry.

            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self._ptm_matrix_values(x)
        bounds_ineq = []
        n = len(self.Gs)
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            i += 1
            for _ in range(ds - 1):
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                bounds_ineq.append(ptm_matrix[i] + 1)  # G_k[i] >= -1
                bounds_ineq.append(-ptm_matrix[i] + 1)  # G_k[i] <= 1
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                i += 1
        return bounds_ineq

    def _rho_trace_constraint(self, x: np.array) -> List[float]:
        """The constraint Tr(rho) = 1
        Args:
            x: The vector representation of the GST data

        Return:
            The list of computed constraint values (should be equal 0)

        Additional information:
            We demand real(Tr(rho)) == 1 and imag(Tr(rho)) == 0
        """
        trace = self._rho_trace(x)
        return [trace[0] - 1, trace[1]]

    def _constraints(self) -> List[Dict]:
        """Generates the constraints for the MLE optimization

        Returns:
            A list of constraints.

        Additional information:
            Each constraint is a dictionary containing
            type ('eq' for equality == 0, 'ineq' for inequality >= 0)
            and a function generating from the input x the values
            that are being constrained.
        """
        cons = []
        cons.append({'type': 'eq', 'fun': self._rho_trace_constraint})
        cons.append({'type': 'eq', 'fun': self._bounds_eq_constraint})
        cons.append({'type': 'ineq', 'fun': self._bounds_ineq_constraint})
        return cons

    def _convert_from_ptm(self, vector):
        """Converts a vector back from PTM representation"""

        pauli_strings_matrices = Pauli_strings(self.num_qubits)
        v = vector.reshape(np.size(vector))
        n = [a * b for a, b in zip(v, pauli_strings_matrices)]
        return reduce(lambda x, y: np.add(x, y), n)

    def _process_result(self, x: np.array) -> Dict:
        """Completes and transforms the optimization result to a friendly format satisfying the physical constraints
        Args:
            x: the optimization result vector

        Returns:
            The final GST data, as dictionary.
        """

        xfinal = self._complete_x(x)

        E, rho, G_matrices = self._split_input_vector(xfinal, 'full')

        result = {}
        result['E'] = Operator(self._convert_from_ptm(E))
        result['rho'] = DensityMatrix(self._convert_from_ptm(rho))
        for i in range(len(self.Gs)):
            #If not exactly PSD, find the closest PSD
            Choi_matrix=Choi(make_positive_semidefinite(Choi(PTM(G_matrices[i])).data))
            gate_matrix=PTM(Choi_matrix).data
            #make it TP if it is not exactly TP (but always will almost TP up to a very small deviation)
            gate_matrix[0][0]=1
            result[self.Gs[i]]= PTM(gate_matrix)
        return result

    def set_initial_value(self, initial_gateset: Dict[str, PTM]):
        """Sets the initial value for the MLE optimization
        Args:
            initial_value: The dictionary of the initial gateset
        """
        E = initial_gateset['E']
        rho = initial_gateset['rho']
        Gs = [initial_gateset[label] for label in self.Gs]

        initial_value_temp = self._join_input_vector(E, rho, Gs)
        n = len(Gs)
        d = (2 ** self.num_qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        m = 0
        for i in range(n):
            # Deleting the final elements of the cholesky decomposition matrices _T of the gates from the
            # initial_value_temp vector (containing the vectorization of GST data).
            initial_value_temp = np.delete(initial_value_temp, 4 * ds + (2 * i + 1) * ds ** 2 - 1 - m)
            m += 1
        return initial_value_temp


    def optimize(self) -> Dict:
        """Performs the MLE optimization for gate set tomography
        Args:
            initial_value: Vector representation of the initial value data
        Returns:
            The formatted results of the MLE optimization.
        """
        result = opt.minimize(self._obj_fn, self.initial_value,
                              method='SLSQP',
                              constraints=self._constraints())
        formatted_result = self._process_result(result.x)
        return formatted_result

