import pennylane as qml
from pennylane import numpy as pnp
from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

BOHR_PER_ANG = 1.8897259886

class NOFVQE:
    @staticmethod
    def _read_mol(filepath):
        symbols, geometry = [], []
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        charge, multiplicity = map(int, lines[0].split()[:2])
        for line in lines[1:]:
            s, x, y, z = line.split()[:4]
            symbols.append(s)
            geometry.append([float(x), float(y), float(z)])
        return charge, multiplicity, symbols, pnp.array(geometry, requires_grad=False)

    @staticmethod
    def _get_no_on(rdm1, norb):
        rdm1_aa = jnp.zeros((norb, norb))
        i = -1
        for p in range(norb):
            for q in range(p, norb):
                i += 1
                rdm1_aa = rdm1_aa.at[p, q].set(rdm1[i])
                rdm1_aa = rdm1_aa.at[q, p].set(rdm1[i])
        n, vecs = jnp.linalg.eigh(rdm1_aa)
        return n[::-1], vecs[:, ::-1]  # sort desc

    def __init__(self, geometry, functional="PNOF4", conv_tol=1e-7, init_param=0.1, max_iterations=1000):
        self.charge, self.mul, self.symbols, self.crd = self._read_mol(geometry)  # Å
        self.functional = functional
        self.conv_tol = conv_tol
        self.init_param = init_param
        self.max_iterations = max_iterations

    # ---------------- Ansatz ----------------
    def _ansatz(self, params, hf_state, qubits):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.DoubleExcitation(params, wires=[0, 1, 2, 3])

    def _build_rdm1_ops(self, norb):
        ops = []
        for p in range(norb):
            for q in range(p, norb):
                cpaq = jordan_wigner(0.5 * (FermiC(2*p) * FermiA(2*q) + FermiC(2*q) * FermiA(2*p))).simplify()
                coeffs = jnp.real(jnp.array(cpaq.terms()[0]))
                obs = cpaq.terms()[1]
                op = coeffs[0] * obs[0]
                for c, o in zip(coeffs[1:], obs[1:]):
                    op += c * o
                ops.append(op)
        return ops

    # ---------- integrals at a geometry (MO basis) ----------
    def _mo_integrals(self, crds_ang):
        """Return (E_nuc, h_MO, I_MO, n_electrons, norb) at given geometry (Å)."""
        mol = qml.qchem.Molecule(self.symbols, crds_ang * BOHR_PER_ANG, unit="bohr")
        core, h_MO, I_MO = qml.qchem.electron_integrals(mol)()  # MO integrals
        E_nuc = core[0]
        n_elec = mol.n_electrons
        norb = int(h_MO.shape[0])
        return jnp.array(E_nuc), jnp.array(h_MO), jnp.array(I_MO), n_elec, norb

    # ---------- measure 1-RDM on the circuit ----------
    def _rdm1_from_circuit(self, params, n_elec, norb):
        qubits = 2 * norb
        hf_state = [1] * n_elec + [0] * (qubits - n_elec)
        dev = qml.device("lightning.qubit", wires=qubits)

        @qml.qnode(dev, interface="jax")
        def rdm1_qnode(theta):
            self._ansatz(theta, hf_state, qubits)
            return [qml.expval(op) for op in self._build_rdm1_ops(norb)]

        return jnp.array(rdm1_qnode(params))

    # ---------- PNOF4 energy (kept as you had it; depends on params, crds) ----------
    def ene_pnof4(self, params, crds_ang):
        E_nuc, h_MO, I_MO, n_elec, norb = self._mo_integrals(crds_ang)
        F = n_elec // 2
        rdm1_vals = self._rdm1_from_circuit(params, n_elec, norb)
        n, C_no_mo = self._get_no_on(rdm1_vals, norb)   # NO coeffs in MO basis
        h = 1 - n
        S_F = jnp.sum(n[F:])

        # MO -> NO transforms for integrals
        h_NO = jnp.einsum("ij,ip,jq->pq", h_MO, C_no_mo, C_no_mo, optimize=True)
        J_NO = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_MO, C_no_mo, C_no_mo, C_no_mo, C_no_mo, optimize=True)
        K_NO = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_MO, C_no_mo, C_no_mo, C_no_mo, C_no_mo, optimize=True)

        # PNOF4 Δ and Π
        Delta = jnp.zeros((norb, norb))
        Pi = jnp.zeros((norb, norb))
        for p in range(norb):
            for q in range(norb):
                if p < F and q < F:
                    Delta = Delta.at[q, p].set(h[q]*h[p])
                    Pi    = Pi.at[q, p].set(-jnp.sqrt(jnp.abs(h[q]*h[p])))
                elif p < F <= q:
                    Delta = Delta.at[q, p].set((1-S_F)/S_F * n[q] * h[p])
                    Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((n[q]*h[p]/S_F) * (n[p]-n[q] + n[q]*h[p]/S_F))))
                elif q < F <= p:
                    Delta = Delta.at[q, p].set((1-S_F)/S_F * h[q] * n[p])
                    Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((h[q]*n[p]/S_F) * (n[q]-n[p] + h[q]*n[p]/S_F))))
                else:
                    Delta = Delta.at[q, p].set(n[q]*n[p])
                    Pi    = Pi.at[q, p].set(jnp.sqrt(jnp.abs(n[q]*n[p])))

        E1 = jnp.sum(2*n * jnp.diag(h_NO)) + jnp.sum(n * jnp.diag(J_NO))
        E2 = 0.0
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    E2 += (n[q]*n[p] - Delta[q, p]) * (2*J_NO[p, q] - K_NO[p, q]) + Pi[q, p] * K_NO[p, q]
        return E_nuc + E1 + E2

    # ---------- VQE for params at fixed geometry ----------
    def _vqe(self, E_fn, params, crds_ang):
        opt = optax.sgd(learning_rate=0.1)
        opt_state = opt.init(params)
        E_single = lambda p: E_fn(p, crds_ang)
        E_history, params_history = [E_single(params)], [params]
        for it in range(self.max_iterations):
            g = jax.grad(E_single)(params)
            updates, opt_state = opt.update(g, opt_state)
            params = optax.apply_updates(params, updates)
            params_history.append(params)
            E_history.append(E_single(params))
            gmax = jnp.max(jnp.abs(g))
            print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {gmax:.1e}")
            if gmax <= self.conv_tol:
                break
        return E_history, params_history

    def ene_vqe(self):
        return self._vqe(self.ene_pnof4, self.init_param, self.crd)

    # ---------- Fedorov-style nuclear gradient ----------
    def nuclear_gradient_fedorov(self, params, crds_ang, h=1.0e-3):
        """
        Compute ∂E/∂R_Aα using the Fedorov idea:
        - Measure RDMs once from the circuit at (params, crds_ang).
        - Recompute **MO** integrals at shifted geometries (R±h e_{Aα}) only.
        - Contract derivative integrals with γ (MO) and Γ (MO).
        h is in **bohr**.
        """
        # Base integrals (to get n_elec, norb) and RDMs
        E_nuc_0, h_MO_0, I_MO_0, n_elec, norb = self._mo_integrals(crds_ang)
        F = n_elec // 2

        # 1-RDM from circuit at the base geometry (only once)
        rdm1_vals = self._rdm1_from_circuit(params, n_elec, norb)
        n, C_no_mo = self._get_no_on(rdm1_vals, norb)
        h_occ = 1.0 - n
        S_F = jnp.sum(n[F:])

        # Build Δ, Π in NO basis (same as in energy)
        Delta = jnp.zeros((norb, norb))
        Pi = jnp.zeros((norb, norb))
        for p in range(norb):
            for q in range(norb):
                if p < F and q < F:
                    Delta = Delta.at[q, p].set(h_occ[q]*h_occ[p])
                    Pi    = Pi.at[q, p].set(-jnp.sqrt(jnp.abs(h_occ[q]*h_occ[p])))
                elif p < F <= q:
                    Delta = Delta.at[q, p].set((1-S_F)/S_F * n[q] * h_occ[p])
                    Pi    = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((n[q]*h_occ[p]/S_F) * (n[p]-n[q] + n[q]*h_occ[p]/S_F))))
                elif q < F <= p:
                    Delta = Delta.at[q, p].set((1-S_F)/S_F * h_occ[q] * n[p])
                    Pi    = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((h_occ[q]*n[p]/S_F) * (n[q]-n[p] + h_occ[q]*n[p]/S_F))))
                else:
                    Delta = Delta.at[q, p].set(n[q]*n[p])
                    Pi    = Pi.at[q, p].set(jnp.sqrt(jnp.abs(n[q]*n[p])))

        # γ and Γ in NO basis
        gamma_NO = jnp.diag(n)
        Gamma_NO = jnp.zeros((norb, norb, norb, norb))
        for p in range(norb):
            for q in range(norb):
                Gamma_NO = Gamma_NO.at[p, q, p, q].set(n[p]*n[q] - Delta[q, p])
                Gamma_NO = Gamma_NO.at[p, q, q, p].set(Pi[q, p])

        # Back-transform to **MO** basis (to match MO integrals)
        C = C_no_mo
        gamma_MO = jnp.einsum("pa,ab,qb->pq", C, gamma_NO, C, optimize=True)
        Gamma_MO = jnp.einsum("ab,cd,pa,qb,rc,sd->pqrs",
                              jnp.eye(norb), jnp.eye(norb), C, C, C, C,
                              optimize=True)  # build tensor map
        # The above line just shows structure; we actually want:
        Gamma_MO = jnp.einsum("abcd,pa,qb,rc,sd->pqrs", Gamma_NO, C, C, C, C, optimize=True)

        # Loop over atoms and Cartesian components; central-diff integral derivatives
        nat = crds_ang.shape[0]
        grad = jnp.zeros((nat, 3))

        for A in range(nat):
            for xyz in range(3):
                # Shift of size h (bohr) converted to Å for the input coordinates
                dR_ang = h / BOHR_PER_ANG

                crds_p = crds_ang.at[A, xyz].add(dR_ang)
                crds_m = crds_ang.at[A, xyz].add(-dR_ang)

                E_nuc_p, h_MO_p, I_MO_p, *_ = self._mo_integrals(crds_p)
                E_nuc_m, h_MO_m, I_MO_m, *_ = self._mo_integrals(crds_m)

                dh = (h_MO_p - h_MO_m) / (2.0 * h)     # ∂h_MO/∂R_{Axyz}
                dI = (I_MO_p - I_MO_m) / (2.0 * h)     # ∂(pq|rs)/∂R_{Axyz}
                dE_nuc = (E_nuc_p - E_nuc_m) / (2.0 * h)

                term1 = jnp.einsum("pq,pq->", gamma_MO, dh, optimize=True)
                term2 = 0.5 * jnp.einsum("pqrs,pqrs->", Gamma_MO, dI, optimize=True)

                grad = grad.at[A, xyz].set(term1 + term2 + dE_nuc)

        return grad

if __name__ == "__main__":
    cal = NOFVQE("h2.xyz", functional="PNOF4", conv_tol=1e-7, init_param=0.1)

    # 1) Optimize the parameter at fixed geometry
    E_hist, p_hist = cal.ene_vqe()
    E_min, params_opt = E_hist[-1], p_hist[-1]
    print("Min Ene VQE and param:", E_min, params_opt)

    # 2) Fedorov-style gradient: reuse RDMs, shift **integrals** only
    grad = cal.nuclear_gradient_fedorov(params_opt, cal.crd, h=1.0e-3)  # h in bohr
    print("Nuclear gradient (Ha/bohr):\n", grad)
    print("Forces (Ha/bohr):\n", -grad)

