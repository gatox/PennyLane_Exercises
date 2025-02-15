import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device("default.qubit", wires=1)

@qml.qnode(dev1, interface="autograd")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

print(circuit([0.54, 0.12]))
