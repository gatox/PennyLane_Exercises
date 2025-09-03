import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import jax
import numpy as np


class NOFVQE:
    def __init__(self, mol_file):
        self.charge, self.mult, self.symbols, self.crd = self._read_mol(mol_file)

    def _read_mol(self, mol_file):
        """Read .xyz file and return charge, multiplicity, symbols, geometry (as jnp.array)."""
        with open(mol_file, "r") as f:
            lines = f.readlines()
        natoms = int(lines[0])
        charge, multiplicity = 0, 1
        symbols, geometry = [], []
        for line in lines[2 : 2 + natoms]:
            tokens = line.split()
            symbols.append(tokens[0])
            geometry.append([float(x) for x in tokens[1:4]])
        return charge, multiplicity, symbols, jnp.array(geometry)  # JAX array!

    def _ansatz(self, params, hf_state, qubits):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.RY(params, wires=0)

    def _build_rdm1_ops(self, norb):
        """Construct operators for 1-RDM measurement."""
        ops = []
        for p in range(norb):
            for q in range(norb):
                ops.append(qml.FermionicOp({((p, 1), (q, 0)): 1.0}, wires=range(2 * norb)).sparse_matrix())
        return ops

    def _get_no_on(self, rdm1_vals, norb):
        """Return natural occupations and orbitals from 1-RDM eigen-decomposition."""
        rdm1 = jnp.array(rdm1_vals).reshape((norb, norb))
        occ, vecs = jnp.linalg.eigh(rdm1)
        idx = jnp.argsort(-occ)  # descending order
        return occ[idx], vecs[:, idx]

    def ene_pnof4(self, params, crds=None):
        """Compute PNOF4 energy functional."""
        if crds is None:
            crds = self.crd

        crds_bohr = crds * 1.8897259886
        mol = qml.qchem.Molecule(self.symbols, crds_bohr, charge=self.charge,
                                 mult=self.mult, unit="bohr")
        core, h_MO, I_MO = qml.qchem.electron_integrals(mol)()
        n_electrons = mol.n_electrons
        norb = h_MO.shape[0]
        qubits = 2 * norb
        F = n_electrons // 2
        hf_state = [1] * n_electrons + [0] * (qubits - n_electrons)

        dev = qml.device("lightning.qubit", wires=qubits)

        @qml.qnode(dev)
        def rdm1_circuit(p):
            self._ansatz(p, hf_state, qubits)
            rdm1_ops = self._build_rdm1_ops(norb)
            return [qml.expval(op) for op in rdm1_ops]

        rdm1_vals = jnp.array(rdm1_circuit(params))
        n, vecs = self._get_no_on(rdm1_vals, norb)

        h_NO = jnp.einsum("ij,ip,jq->pq", h_MO, vecs, vecs)
        J_NO = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_MO, vecs, vecs, vecs, vecs)
        K_NO = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_MO, vecs, vecs, vecs, vecs)

        # build PNOF4 matrices
        h = 1 - n
        S_F = jnp.sum(n[F:])
        Delta = jnp.zeros((norb, norb))
        Pi = jnp.zeros((norb, norb))
        for p in range(norb):
            for q in range(norb):
                if p < F and q < F:
                    Delta = Delta.at[q, p].set(h[q] * h[p])
                    Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs(h[q] * h[p])))
                elif p < F and q >= F:
                    Delta = Delta.at[q, p].set((1 - S_F) / S_F * n[q] * h[p])
                    Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) *
                                                           (n[p] - n[q] + n[q] * h[p] / S_F))))
                elif p >= F and q < F:
                    Delta = Delta.at[q, p].set((1 - S_F) / S_F * h[q] * n[p])
                    Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) *
                                                           (n[q] - n[p] + h[q] * n[p] / S_F))))
                else:
                    Delta = Delta.at[q, p].set(n[q] * n[p])
                    Pi = Pi.at[q, p].set(jnp.sqrt(jnp.abs(n[q] * n[p])))

        E1 = 2 * jnp.sum(n * jnp.diag(h_NO)) + jnp.sum(n * jnp.diag(J_NO))
        E2 = 0.0
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    E2 += (n[q] * n[p] - Delta[q, p]) * (2 * J_NO[p, q] - K_NO[p, q])
                    E2 += Pi[q, p] * K_NO[p, q]
        return E1 + E2

    def nuclear_gradient_fedorov(self, params, crds, h=1.0e-3):
        """Fedorov-style nuclear gradient: shift integrals, reuse RDMs."""
        crds_bohr = crds * 1.8897259886
        natoms, _ = crds_bohr.shape
        grad = jnp.zeros_like(crds_bohr)

        # Reference mol + NO basis
        mol0 = qml.qchem.Molecule(self.symbols, crds_bohr, unit="bohr")
        core, h_MO, I_MO = qml.qchem.electron_integrals(mol0)()
        n_electrons = mol0.n_electrons
        norb = h_MO.shape[0]
        qubits = 2 * norb
        F = n_electrons // 2
        hf_state = [1] * n_electrons + [0] * (qubits - n_electrons)

        dev = qml.device("lightning.qubit", wires=qubits)

        @qml.qnode(dev)
        def rdm1_circuit(p):
            self._ansatz(p, hf_state, qubits)
            rdm1_ops = self._build_rdm1_ops(norb)
            return [qml.expval(op) for op in rdm1_ops]

        rdm1_vals = jnp.array(rdm1_circuit(params))
        n, vecs = self._get_no_on(rdm1_vals, norb)

        def ene_from_ints(h_NO, J_NO, K_NO, n, F):
            h = 1 - n
            S_F = jnp.sum(n[F:])
            Delta = jnp.zeros((norb, norb))
            Pi = jnp.zeros((norb, norb))
            for p in range(norb):
                for q in range(norb):
                    if p < F and q < F:
                        Delta = Delta.at[q, p].set(h[q] * h[p])
                        Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs(h[q] * h[p])))
                    elif p < F and q >= F:
                        Delta = Delta.at[q, p].set((1 - S_F) / S_F * n[q] * h[p])
                        Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) *
                                                               (n[p] - n[q] + n[q] * h[p] / S_F))))
                    elif p >= F and q < F:
                        Delta = Delta.at[q, p].set((1 - S_F) / S_F * h[q] * n[p])
                        Pi = Pi.at[q, p].set(-jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) *
                                                               (n[q] - n[p] + h[q] * n[p] / S_F))))
                    else:
                        Delta = Delta.at[q, p].set(n[q] * n[p])
                        Pi = Pi.at[q, p].set(jnp.sqrt(jnp.abs(n[q] * n[p])))

            E1 = 2 * jnp.sum(n * jnp.diag(h_NO)) + jnp.sum(n * jnp.diag(J_NO))
            E2 = 0.0
            for p in range(norb):
                for q in range(norb):
                    if p != q:
                        E2 += (n[q] * n[p] - Delta[q, p]) * (2 * J_NO[p, q] - K_NO[p, q])
                        E2 += Pi[q, p] * K_NO[p, q]
            return E1 + E2

        # finite-difference loop
        for A in range(natoms):
            for xyz in range(3):
                dR = jnp.zeros_like(crds_bohr)
                dR = dR.at[A, xyz].set(h)
                crds_p, crds_m = crds_bohr + dR, crds_bohr - dR

                mol_p = qml.qchem.Molecule(self.symbols, crds_p, unit="bohr")
                mol_m = qml.qchem.Molecule(self.symbols, crds_m, unit="bohr")
                _, h_p, I_p = qml.qchem.electron_integrals(mol_p)()
                _, h_m, I_m = qml.qchem.electron_integrals(mol_m)()

                h_p_NO = jnp.einsum("ij,ip,jq->pq", h_p, vecs, vecs)
                h_m_NO = jnp.einsum("ij,ip,jq->pq", h_m, vecs, vecs)
                J_p = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_p, vecs, vecs, vecs, vecs)
                J_m = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_m, vecs, vecs, vecs, vecs)
                K_p = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_p, vecs, vecs, vecs, vecs)
                K_m = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_m, vecs, vecs, vecs, vecs)

                E_p = ene_from_ints(h_p_NO, J_p, K_p, n, F)
                E_m = ene_from_ints(h_m_NO, J_m, K_m, n, F)

                grad = grad.at[A, xyz].set((E_p - E_m) / (2 * h))

        return grad / 1.8897259886  # back to Å

if __name__ == "__main__":
    cal = NOFVQE("h2.xyz")
    energy = cal.ene_pnof4(params=0.2261)
    grad = cal.nuclear_gradient_fedorov(params=0.2261, crds=cal.crd, h=1.0e-3)
    print("Energy:", energy)
    print("Gradient:\n", grad)
