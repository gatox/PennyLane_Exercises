import pennylane as qml
from pennylane import numpy as pnp

from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp

import optax

jax.config.update("jax_enable_x64", True)

class NOFVQE():

    @staticmethod
    def _read_mol(filepath):
        """
        Reads a simple XYZ-like file that starts with charge/multiplicity
        and returns charge, multiplicity, symbols, and geometry.

        Parameters:
            filepath (str): Path to the input file.

        Returns:
            charge (int): Molecular charge.
            multiplicity (int): Spin multiplicity.
            symbols (list[str]): Atomic symbols.
            geometry (list[list[float]]): Atomic coordinates.
        """
        symbols = []
        geometry = []

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # First line: charge and multiplicity
        parts = lines[0].split()
        charge, multiplicity = int(parts[0]), int(parts[1])

        # Remaining lines: atoms
        for line in lines[1:]:
            parts = line.split()
            symbol = parts[0]
            x, y, z = map(float, parts[1:4])
            symbols.append(symbol)
            geometry.append([x, y, z])

        return charge, multiplicity, symbols, pnp.array(geometry,requires_grad=False)

     
    @staticmethod
    def _get_no_on(rdm1, norb):
    
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


    def __init__(self, geometry, functional="PNOF4", conv_tol=1e-7, init_param=0.1, max_iterations = 1000):
        self.charge, self.mul, self.symbols, self.crd = self._read_mol(geometry)
        self.functional = functional
        self.conv_tol = conv_tol
        self.init_param = init_param
        self.max_iterations = max_iterations
        #self.mol = qml.qchem.Molecule(self.symbols, self.crd * 1.8897259886, unit="bohr")
        #self.n_electrons = self.mol.n_electrons

            # ---------------- Ansätze ----------------
    def _ansatz(self, params, hf_state, qubits):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.DoubleExcitation(params, wires=[0, 1, 2, 3])

    def _build_rdm1_ops(self, norb):
        ops = []
        for p in range(0, norb):
            for q in range(p, norb):
                cpaq = jordan_wigner(
                    0.5 * (FermiC(2 * p) * FermiA(2 * q) + FermiC(2 * q) * FermiA(2 * p))
                ).simplify()
                #### everything is real by construction
                #### cast coefficients to jax reals to avoid warnings with zero values in imaginary parts
                coeffs = jnp.real(jnp.array(cpaq.terms()[0]))
                obs = cpaq.terms()[1]
                cpaq = coeffs[0] * obs[0]
                for coeff, op in zip(coeffs[1:], obs[1:]):
                    cpaq += coeff * op
                ops.append(cpaq)
        return ops

    def ene_pnof4(self, params, crds):
        # Functions based on 1-RDM (J. Chem. Theory Comput. 2025, 21, 5, 2402–2413) and taked from the following repository:
        # https://github.com/felipelewyee/NOF-VQE
        
        # Build molecule and integrals dependent on geometry
        mol = qml.qchem.Molecule(self.symbols, crds * 1.8897259886, unit="bohr")
        core, h_MO, I_MO = qml.qchem.electron_integrals(mol)()
        n_electrons = mol.n_electrons

        E_nuc = core[0]
        
        norb = pnp.shape(h_MO)[0]
        qubits = 2 * norb
        
        # Fermi level
        F = int(n_electrons / 2)
        
        hf_state = [1] * n_electrons + [0] * (qubits - n_electrons)
        #
        dev = qml.device("lightning.qubit", wires=qubits)
        @qml.qnode(dev)
        def rdm1_circuit(params, hf_state, qubits, norb):
            self._ansatz(params, hf_state, qubits)
            rdm1_ops = self._build_rdm1_ops(norb)
            return [qml.expval(op) for op in rdm1_ops]

        rdm1_vals = rdm1_circuit(params, hf_state, qubits, norb)
        rdm1_vals = jnp.array(rdm1_vals)   # ensure JAX array
        n, vecs = self._get_no_on(rdm1_vals,norb)
        h = 1 - n
        S_F = jnp.sum(n[F:])

    
        h_NO = jnp.einsum("ij,ip,jq->pq", h_MO, vecs, vecs, optimize=True)
        J_NO = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)
        K_NO = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)
    
    
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
                    val = -jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) * (n[p] - n[q] + n[q] * h[p] / S_F)))
                if p >= F and q < F:
                    val = -jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) * (n[q] - n[p] + h[q] * n[p] / S_F)))
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
   # Energy minimization
   # =========================

    def _vqe(self, E_fn, params, crds):
        opt = optax.sgd(learning_rate=0.1)
        opt_state = opt.init(params)
        
        # energy function that depends only on params (geometry fixed)
        E_single = lambda p: E_fn(p, crds)
        
        E_history = [E_single(params)]
        params_history = [params]
        
        for it in range(self.max_iterations):
        
            gradient = jax.grad(E_single)(params)
        
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
        
            params_history.append(params)
            E_history.append(E_single(params))
        
            g_maxabs = jnp.max(jnp.abs(gradient))
        
            print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
        
            if g_maxabs <= self.conv_tol:
                break
        
        return E_history, params_history


    def nuclear_gradient(self, params, crds):
        """Compute nuclear gradient dE/dR at fixed params."""
        return jax.grad(lambda c: self.ene_pnof4(params, c))(crds)


    #def _vqe(self, E_fn, params, crds):
    #    opt = optax.sgd(learning_rate=0.1)
    #    opt_state = opt.init(params)
    #
    #    E_history = [E_fn(params, crds)]
    #    params_history = [params]
    #
    #    for it in range(self.max_iterations):
    #
    #        gradient = jax.grad(E_fn)(params)
    #
    #        updates, opt_state = opt.update(gradient, opt_state)
    #        params = optax.apply_updates(params, updates)
    #
    #        params_history.append(params)
    #        E_history.append(E_fn(params))
    #
    #        g_maxabs = jnp.max(jnp.abs(gradient))
    #
    #        print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
    #
    #        if g_maxabs <= self.conv_tol:
    #            break
    #
    #    return E_history, params_history

    def ene_vqe(self):
        E_history, params_history = self._vqe(self.ene_pnof4, self.init_param, self.crd)
        return E_history, params_history



# =========================
# Run the calculation
# =========================
if __name__ == "__main__":

    cal = NOFVQE("h2.xyz", functional="PNOF4", conv_tol=1e-7, init_param=0.1)

    # Run VQE
    E_history, params_history = cal.ene_vqe()
    E_min, params_opt = E_history[-1], params_history[-1]
    print("Min Ene VQE and param:", E_min, params_opt)

    ## Nuclear gradient at optimized geometry
    #grad_R = cal.nuclear_gradient(params_opt, cal.crd)
    #print("Nuclear gradient:\n", grad_R)

#if __name__ == "__main__":
#
#    # Initial conditions
#    geo = "h2.xyz"
#    functional = "PNOF4" 
#    conv_tol = 1e-7
#    init_param = 0.1
#    max_iterations = 1000
#
#    # Calling the class
#    cal = NOFVQE(geo, functional, conv_tol, init_param, max_iterations)
#    print(cal.crd)
#    print(cal.symbols)
#    #print(cal.n_electrons)
#    #print(cal.ene_pnof4(params=0.22501))
#    
#    #grad = jax.grad(E_PNOF4)(params)
#
#    E_history, params_history = cal.ene_vqe()
#    print("Min Ene VQE and param:",E_history[-1], params_history[-1])
