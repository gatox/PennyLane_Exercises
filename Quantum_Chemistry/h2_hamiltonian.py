import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H","H"]
coordinates = np.array([[-0.673, 0, 0],[0.673, 0, 0]])

H, qubits = qchem.molecular_hamiltonian(symbols,coordinates)
#print(qubits)
#print(H)

#Quantum circuit to compute the expectation value for an energy
#state in the Jordan-Wigner representation

num_wires = qubits
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def expec_energy(state):
    qml.BasisState(np.array(state),wires=range(num_wires))
    return qml.expval(H)

#val = expec_energy([1,0,1,0])
#print(expec_energy([1,0,1,0]))
#Hartree_Fock state
hf = qchem.hf_state(electrons=2,orbitals=4)
print("Hartree_Fock State:", hf)
print("Expectation Energy Value:",expec_energy(hf))
