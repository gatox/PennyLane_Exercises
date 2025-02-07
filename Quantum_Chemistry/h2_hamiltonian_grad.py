import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from pennylane.qchem import molecular_hamiltonian, decompose_hamiltonian
from pennylane.gradients import finite_diff, param_shift

# Define the molecular system (H2)
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])  # Atomic positions in Bohr

# Generate the molecular Hamiltonian
H, qubits = molecular_hamiltonian(symbols, coordinates, charge=0)
print("Number of qubits:", qubits)
print("Hamiltonian:", H)

# Define the ansatz
def ansatz(params):
    qml.BasisState(np.array([1, 1]), wires=[0, 1])  # Hartree-Fock reference state
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=1)

# Define the device and cost function
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H)

# Optimize the parameters
init_params = np.random.rand(3)  # Random initial parameters
result = minimize(cost_fn, init_params, method="COBYLA")
optimal_params = result.x

print("Optimized Parameters:", optimal_params)
print("Estimated Ground-State Energy:", cost_fn(optimal_params))

# Compare with classical Full Configuration Interaction (FCI) energy
H_matrix = decompose_hamiltonian(H)
eigenvalues = np.linalg.eigvals(H_matrix)
fci_energy = min(eigenvalues)

print("FCI Energy:", fci_energy)
print("VQE Error:", abs(fci_energy - cost_fn(optimal_params)))

# Compute nuclear gradients using different methods
# 1. Finite Difference Method
def nuclear_gradient_finite_diff(params):
    grad_fn = finite_diff(cost_fn, argnum=0)
    return grad_fn(params)

# 2. Parameter-Shift Rule
def nuclear_gradient_param_shift(params):
    grad_fn = param_shift(cost_fn)
    return grad_fn(params)

# 3. Automatic Differentiation
def nuclear_gradient_autograd(params):
    grad_fn = qml.grad(cost_fn)
    return grad_fn(params)

# Compute the gradients and forces
grad_finite_diff = nuclear_gradient_finite_diff(optimal_params)
grad_param_shift = nuclear_gradient_param_shift(optimal_params)
grad_autograd = nuclear_gradient_autograd(optimal_params)

forces_finite_diff = -grad_finite_diff
forces_param_shift = -grad_param_shift
forces_autograd = -grad_autograd

print("Nuclear Forces (Finite Difference):", forces_finite_diff)
print("Nuclear Forces (Parameter-Shift):", forces_param_shift)
print("Nuclear Forces (AutoDiff):", forces_autograd)

# Potential Interface with PySurf for AIMD
# The computed nuclear forces can be fed into PySurf for ground-state ab initio dynamics

