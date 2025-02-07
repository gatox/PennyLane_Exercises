import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import finite_diff

# Define the device
dev = qml.device("default.qubit", wires=2)

# Define the ansatz
def ansatz(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

# Define multiple cost functions (for different states)
@qml.qnode(dev)
def cost_ground(params):
    ansatz(params, wires=[0, 1])
    return qml.expval(H)

@qml.qnode(dev)
def cost_excited(params):
    qml.PauliX(wires=0)  # Apply X-gate for state variation
    ansatz(params, wires=[0, 1])
    return qml.expval(H)

# Optimize multiple states
opt = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.random.rand(2)

for i in range(100):
    params = opt.step(cost_ground, params)  # Optimize ground state
    params = opt.step(cost_excited, params)  # Optimize excited state

print("Ground-State Energy:", cost_ground(params))
print("Excited-State Energy:", cost_excited(params))

#Excited states gradient

def excited_state_gradient(params):
    grad_fn = finite_diff(cost_excited, argnum=0)  # Differentiate excited-state energy
    return grad_fn(params)

grad_excited = excited_state_gradient(params)
forces_excited = -grad_excited
print("Excited-State Nuclear Forces:", forces_excited)
