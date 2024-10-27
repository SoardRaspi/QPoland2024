import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer
import pylatexenc

class VQLSSolver:
    def __init__(self, n_qubits, n_layers, kappa=10):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kappa = kappa
        self.training_history = []

    def create_ansatz(self, parameters, ans_type):
      """Hardware-efficient ansatz for the VQLS algorithm with CX gates"""
      qc = QuantumCircuit(self.n_qubits)
      param_idx = 0

      # print("ans_type in create_ansatz:", ans_type)

      if ans_type == "standard":
        # Initial rotation layer
        for i in range(self.n_qubits):
            qc.ry(parameters[param_idx], i)
            param_idx += 1

        for layer in range(self.n_layers):
            # CX layer on alternating pairs
            for i in range(0, self.n_qubits-1, 2):
                qc.cx(i, i+1)
            for i in range(1, self.n_qubits-1, 2):
                qc.cx(i, i+1)

            # Rotation layer
            for i in range(self.n_qubits):
                if param_idx < len(parameters):
                    qc.ry(parameters[param_idx], i)
                    param_idx += 1
        
      if ans_type == "proposed_first":
        # First layer starter
        for i in range(self.n_qubits):
          qc.ry(parameters[param_idx], i)
          param_idx += 1

        for layer_index in range(self.n_layers):
          for i in range(self.n_qubits // 2):
            qc.cz(2*i, 2*i + 1)

          for i in range(self.n_qubits):
            qc.ry(parameters[param_idx], i)
            param_idx += 1

          for i in range(1, self.n_qubits - 1, 2):
            qc.cz(i, i + 1)
          qc.cz(0, self.n_qubits-1)

          for i in range(1, self.n_qubits - 1):
            qc.ry(parameters[param_idx], i)
            param_idx += 1

      return qc

    def build_hamiltonian(self):
        """Build A matrix as per the paper's specification"""
        zeta = 1/np.sqrt(self.n_qubits + 0.1 * (self.n_qubits - 1))  # Normalization
        J = 0.1  # As specified in paper

        # Pauli operator strings
        paulis = []
        coeffs = []

        # Single X terms
        for j in range(self.n_qubits):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[j] = 'X'
            paulis.append(''.join(pauli_str))
            coeffs.append(zeta)

        # ZZ terms
        for j in range(self.n_qubits - 1):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[j] = 'Z'
            pauli_str[j+1] = 'Z'
            paulis.append(''.join(pauli_str))
            coeffs.append(J * zeta)

        return SparsePauliOp(paulis, coeffs)

    def calculate_local_cost(self, parameters, ans_type):
        """Compute the cost function (CL) from equation (6) in the paper"""
        circuit = self.create_ansatz(parameters, ans_type=ans_type)
        x_state = Statevector.from_instruction(circuit)
        A = self.build_hamiltonian()

        # Calculate A|x⟩
        Ax = Statevector(np.dot(A.to_matrix(), x_state.data))
        norm_Ax = np.sqrt(np.abs(np.vdot(Ax.data, Ax.data)))

        # Local cost calculation
        cost = 0
        for j in range(self.n_qubits):
            meas_op = ['I'] * self.n_qubits
            meas_op[j] = 'Z'
            proj_0 = SparsePauliOp([(''.join(meas_op))], [0.5]) + \
                     SparsePauliOp([('I' * self.n_qubits)], [0.5])

            exp_val = np.real(np.vdot(Ax.data,
                             np.dot(proj_0.to_matrix(), Ax.data))) / (norm_Ax ** 2)
            cost += (1 - exp_val) / self.n_qubits

        return float(cost)

    def train(self, ans_type, n_epochs=100):
        """Train the ansatz parameters to minimize the cost function using BFGS"""

        if ans_type == "standard":
          n_params = self.n_qubits * (self.n_layers + 1)
        
        if ans_type == "proposed_first":
          n_params = 2 * (self.n_qubits - 1) * (self.n_layers) + self.n_qubits

        initial_params = np.random.random(n_params) * 2 * np.pi

        # Callback function to store the cost at each epoch
        def callback(xk):
            cost = self.calculate_local_cost(xk, ans_type=ans_type)
            self.training_history.append(cost)
            print(f"Epoch {len(self.training_history)}: Cost = {cost}")  # Print cost per epoch

        result = minimize(
            self.calculate_local_cost,
            initial_params,
            ans_type,
            method='BFGS',
            options={'maxiter': n_epochs},
            callback=callback
        )

        return result

def run_vqls(ans_type):
    # Initialize solver with parameters matching paper example
    n_qubits = 4
    n_layers = 4
    solver = VQLSSolver(n_qubits=n_qubits, n_layers=n_layers)

    # Train and visualize training progress
    result = solver.train(ans_type, n_epochs=200)

    # Draw the final ansatz circuit
    final_circuit = solver.create_ansatz(result.x, ans_type)
    print("Final Ansatz Circuit:")
    display(circuit_drawer(final_circuit, output='mpl', scale=0.8))

    # Plot the cost function over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(solver.training_history)), solver.training_history, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost Function Decrease During Training')
    plt.grid(True)
    plt.show()

    return solver, result

if __name__ == "__main__":
    solver, result = run_vqls("standard")
