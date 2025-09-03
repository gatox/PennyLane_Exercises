import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from pennylane.qchem import molecular_hamiltonian, hf_state

# =========================
# System setup (H2, STO-3G)
# =========================
symbols = ["H", "H"]
# Atomic positions in Bohr (x1,y1,z1, x2,y2,z2)
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614], dtype=float)
charge = 0
electrons = 2

# =========================
# Ansatz
# =========================
def make_cost_fn(H, wires, hf):
    """Build a QNode (cost function) for <H> with a simple 2-qubit ansatz."""
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface="autograd")
    def cost_fn(params):
        qml.BasisState(hf, wires=range(wires))
        # Simple but adequate entangling ansatz for H2 after tapering (2 qubits)
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[2], wires=1)
        return qml.expval(H)

    return cost_fn

def vqe_optimize(H, qubits, params0=None, maxiter=100, tol=1e-7):
    """Optimize VQE parameters for Hamiltonian H, returning (energy, params)."""
    # Hartree–Fock reference state |11> for H2 in STO-3G after mapping (2 electrons)
    hf = hf_state(electrons, qubits)
    hf = np.array(hf, requires_grad=False)

    # Cost function (QNode)
    cost_fn = make_cost_fn(H, qubits, hf)

    # If no initial params given, randomize
    if params0 is None:
        params0 = np.random.uniform(0.0, 2*np.pi, size=3)

    # Use a gradient-free SciPy optimizer (robust for small circuits)
    res = minimize(lambda x: float(cost_fn(x)),
                   x0=np.array(params0, dtype=float),
                   method="COBYLA",
                   options={"maxiter": maxiter, "tol": tol})

    return float(cost_fn(res.x)), res.x

def vqe_energy_at_geometry(coords, params_init=None, maxiter=200):
    """Build H at coords, optimize VQE, return (E, H, qubits, params_opt)."""
    H, qubits = molecular_hamiltonian(symbols, coords, charge=charge)
    E, p_opt = vqe_optimize(H, qubits, params0=params_init, maxiter=maxiter)
    return E, H, qubits, p_opt

# =========================================
# FCI (exact) energy from the Hamiltonian
# =========================================
def exact_ground_energy(H):
    H_dense = qml.matrix(H)
    # np.linalg.eigvals may return complex with ~0 imaginary noise
    evals = np.linalg.eigvals(H_dense)
    evals = np.real_if_close(evals)
    return float(np.min(evals))

# ==============================================================
# Finite-difference nuclear gradient with VQE re-optimization
# ==============================================================
def nuclear_gradient_fd(coords,
                        step=5e-3,          # Bohr (typical 0.002–0.01)
                        maxiter=80,         # inner VQE optimization iterations
                        warm_start=True):
    """
    Central finite-difference gradient dE/dR_i for *nuclear coordinates*.
    For each coordinate, re-optimize VQE at R+h and R-h (warm-started).
    Returns grad (shape (6,)) and forces = -grad.
    """
    coords = np.array(coords, dtype=float)
    assert coords.shape == (6,)

    # First, optimize at the reference geometry
    E0, H0, qubits0, p0 = vqe_energy_at_geometry(coords, params_init=None, maxiter=maxiter)
    print(f"Reference VQE energy: {E0:.12f} Ha  (qubits = {qubits0})")

    grad = np.zeros_like(coords)

    for i in range(len(coords)):
        cp = coords.copy()
        cm = coords.copy()
        cp[i] += step
        cm[i] -= step

        # Warm start from the reference optimal params (or the last one)
        p_start_plus = p0 if warm_start else None
        p_start_minus = p0 if warm_start else None

        Ep, _, _, p_plus = vqe_energy_at_geometry(cp, params_init=p_start_plus, maxiter=maxiter)
        Em, _, _, p_minus = vqe_energy_at_geometry(cm, params_init=p_start_minus, maxiter=maxiter)

        # Central difference
        grad[i] = (Ep - Em) / (2.0 * step)

        # Optional: update warm-start for the next coordinate
        if warm_start:
            p0 = p_plus  # carry forward the latest parameters

        print(f"Coord {i}: Ep={Ep:.12f}, Em={Em:.12f}, dE/dR={grad[i]: .8e} Ha/Bohr")

    forces = -grad
    return grad, forces, E0, H0, qubits0, p0

# =========================
# Run the calculation
# =========================
if __name__ == "__main__":
    # 1) VQE at reference geometry
    E_ref, H_ref, nq, p_ref = vqe_energy_at_geometry(coordinates, maxiter=200)
    print("\n--- VQE at reference geometry ---")
    print(f"Optimized parameters: {p_ref}")
    print(f"Estimated ground-state energy (VQE): {E_ref:.12f} Ha")

    # 2) Exact (FCI) energy for comparison
    E_fci = exact_ground_energy(H_ref)
    print(f"Exact ground-state energy (FCI):   {E_fci:.12f} Ha")
    print(f"VQE error:                         {abs(E_fci - E_ref):.3e} Ha")

    # 3) Nuclear gradients / forces (finite difference with re-optimization)
    print("\n--- Finite-difference nuclear gradients (re-optimized) ---")
    grad, forces, _, _, _, _ = nuclear_gradient_fd(coordinates, step=5e-3, maxiter=80, warm_start=True)

    # Reshape to (2 atoms, 3 Cartesian)
    grad_xyz = grad.reshape(2, 3)
    forces_xyz = forces.reshape(2, 3)

    print("\nNuclear gradients dE/dR (Ha/Bohr):")
    print(grad_xyz)
    print("\nNuclear forces -dE/dR (Ha/Bohr):")
    print(forces_xyz)

