import pennylane as qml
from pennylane import numpy as pnp
from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

# =========================
# System setup
# =========================
symbols = ["H", "H"]
geometry = pnp.array([[0.0, 0.0, 0.0], [0.7414, 0.0, 0.0]], requires_grad=False)

mol = qml.qchem.Molecule(symbols, geometry, unit="angstrom")
core, h_MO, I_MO = qml.qchem.electron_integrals(mol)()
E_nuc = core[0]

norb = pnp.shape(h_MO)[0]
qubits = 2 * norb
electrons = mol.n_electrons
F = int(electrons / 2)

hf_state = [1] * electrons + [0] * (qubits - electrons)

# =========================
# Ansatz
# =========================
def ansatz(params):
    qml.BasisState(hf_state, wires=range(4))
    qml.DoubleExcitation(params, wires=[0, 1, 2, 3])

# =========================
# 1-RDM operators
# =========================
rdm1_ops = []
for p in range(0, norb):
    for q in range(p, norb):
        cpaq = jordan_wigner(
            0.5 * (FermiC(2 * p) * FermiA(2 * q) + FermiC(2 * q) * FermiA(2 * p))
        ).simplify()
        coeffs = jnp.real(jnp.array(cpaq.terms()[0]))
        obs = cpaq.terms()[1]
        cpaq = coeffs[0] * obs[0]
        for coeff, op in zip(coeffs[1:], obs[1:]):
            cpaq += coeff * op
        rdm1_ops.append(cpaq)

dev = qml.device("lightning.qubit", wires=qubits)

@qml.qnode(dev)
def rdm1_circuit(params):
    ansatz(params)
    return [qml.expval(op) for op in rdm1_ops]

# =========================
# Helper: diagonalize 1-RDM
# =========================
def get_no_on(rdm1):
    rdm1_aa = jnp.zeros((norb, norb))
    i = -1
    for p in range(0, norb):
        for q in range(p, norb):
            i = i + 1
            rdm1_aa = rdm1_aa.at[p, q].set(rdm1[i])
            rdm1_aa = rdm1_aa.at[q, p].set(rdm1[i])
    n, vecs = jnp.linalg.eigh(rdm1_aa)
    n = n[::-1]
    vecs = vecs[:, ::-1]
    return n, vecs

# =========================
# PNOF4 energy functional
# =========================
def E_PNOF4(params, rdm1=None):
    if rdm1 is None:
        rdm1 = rdm1_circuit(params)
    n, vecs = get_no_on(rdm1)
    h = 1 - n

    h_NO = jnp.einsum("ij,ip,jq->pq", h_MO, vecs, vecs, optimize=True)
    J_NO = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)
    K_NO = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)

    S_F = jnp.sum(n[F:])

    Delta = jnp.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            val = 0
            if p < F and q < F:
                val = h[q] * h[p]
            if p < F and q >= F:
                val = (1 - S_F) / S_F * n[q] * h[p]
            if p >= F and q < F:
                val = (1 - S_F) / S_F * h[q] * n[p]
            if p >= F and q >= F:
                val = n[q] * n[p]
            Delta = Delta.at[q, p].set(val)

    Pi = jnp.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            val = 0
            if p < F and q < F:
                val = -jnp.sqrt(jnp.abs(h[q] * h[p]))
            if p < F and q >= F:
                val = -jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) *
                                        (n[p] - n[q] + n[q] * h[p] / S_F)))
            if p >= F and q < F:
                val = -jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) *
                                        (n[q] - n[p] + h[q] * n[p] / S_F)))
            if p >= F and q >= F:
                val = jnp.sqrt(jnp.abs(n[q] * n[p]))
            Pi = Pi.at[q, p].set(val)

    E1 = 0
    for p in range(norb):
        E1 += 2 * n[p] * h_NO[p, p]
    for p in range(norb):
        E1 += n[p] * J_NO[p, p]

    E2 = 0
    for p in range(norb):
        for q in range(norb):
            if p != q:
                E2 += (n[q] * n[p] - Delta[q, p]) * (2 * J_NO[p, q] - K_NO[p, q])
                E2 += Pi[q, p] * (K_NO[p, q])

    return E_nuc + E1 + E2

# =========================
# Gradient function (Fedorov et al. 2021)
# =========================
def nuclear_gradient_fedorov(params, geometry_bohr):
    rdm1_vals = rdm1_circuit(params)
    n, vecs = get_no_on(rdm1_vals)
    h = 1 - n
    S_F = jnp.sum(n[F:])

    Delta = jnp.zeros((norb, norb))
    Pi = jnp.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            if p < F and q < F:
                Delta = Delta.at[q, p].set(h[q] * h[p])
                Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs(h[q] * h[p])))
            if p < F and q >= F:
                Delta = Delta.at[q, p].set((1 - S_F) / S_F * n[q] * h[p])
                Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) *
                                                       (n[p] - n[q] + n[q] * h[p] / S_F))))
            if p >= F and q < F:
                Delta = Delta.at[q, p].set((1 - S_F) / S_F * h[q] * n[p])
                Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) *
                                                       (n[q] - n[p] + h[q] * n[p] / S_F))))
            if p >= F and q >= F:
                Delta = Delta.at[q, p].set(n[q] * n[p])
                Pi = Pi.at[q, p].set(jnp.sqrt(jnp.abs(n[q] * n[p])))

    # Build gamma and Gamma in NO basis
    gamma_NO = jnp.diag(n)
    Gamma_NO = jnp.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            Gamma_NO = Gamma_NO.at[p, q, p, q].set(n[p] * n[q] - Delta[q, p])
            Gamma_NO = Gamma_NO.at[p, q, q, p].set(Pi[q, p])

    # Back-transform to AO basis
    C = vecs
    gamma_AO = jnp.einsum("pq,ip,jq->ij", gamma_NO, C, C, optimize=True)
    Gamma_AO = jnp.einsum("pqrs,ip,jq,kr,ls->ijkl", Gamma_NO, C, C, C, C, optimize=True)

    # Get derivative integrals
    mol_tmp = qml.qchem.Molecule(symbols, geometry_bohr, unit="bohr")
    core_tmp, h_AO, I_AO, dh_AO, dI_AO = qml.qchem.electron_integrals(
        mol_tmp, derivatives=True
    )()

    natoms = geometry_bohr.shape[0]
    grad = jnp.zeros((natoms, 3))
    for A in range(natoms):
        for xyz in range(3):
            term1 = jnp.einsum("pq,pq->", gamma_AO, dh_AO[A][xyz], optimize=True)
            term2 = 0.5 * jnp.einsum("pqrs,pqrs->", Gamma_AO, dI_AO[A][xyz], optimize=True)
            grad = grad.at[A, xyz].set(term1 + term2)

    return grad

# =========================
# VQE optimization loop
# =========================
def vqe(E_fn, params, conv_tol, max_iterations):
    opt = optax.sgd(learning_rate=0.1)
    opt_state = opt.init(params)
    E_history = [E_fn(params)]
    params_history = [params]

    for it in range(max_iterations):
        gradient = jax.grad(E_fn)(params)
        updates, opt_state = opt.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)
        params_history.append(params)
        E_history.append(E_fn(params))
        g_maxabs = jnp.max(jnp.abs(gradient))
        print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
        if g_maxabs <= conv_tol:
            break
    return E_history, params_history

# =========================
# Main
# =========================
if __name__ == "__main__":
    max_iterations = 1000
    conv_tol = 1e-7
    ene_hist, params_hist = vqe(E_PNOF4, 0.1, conv_tol, max_iterations)
    params_opt = params_hist[-1]
    print("\nNOFVQE_Energy:", ene_hist[-1])
    print("NOFVQE_Params:", params_opt)

    # Convert geometry to Bohr for gradient
    geometry_bohr = geometry * 1.8897259886  # Å → Bohr
    grad = nuclear_gradient_fedorov(params_opt, geometry_bohr)
    forces = -grad
    print("\nNuclear gradient (Ha/Bohr):")
    print(grad)
    print("\nNuclear forces (Ha/Bohr):")
    print(forces)

