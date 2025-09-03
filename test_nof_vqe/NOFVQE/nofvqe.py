import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp

import optax

jax.config.update("jax_enable_x64", True)

class NOFVQE:

    @staticmethod
    def _read_mol(inputdata):
        """
        Reads either a filepath (XYZ-like file) or a geometry string.
        """
        if "\n" in inputdata or "unit" in inputdata:
            # assume it's a raw string, split into lines
            lines = [line.strip() for line in inputdata.splitlines() if line.strip()]
        else:
            # assume it's a file path
            with open(inputdata, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

        unit = lines[0].split()[1]
        charge, multiplicity = map(int, lines[1].split())

        symbols = []
        geometry = []
        for line in lines[2:]:
            parts = line.split()
            symbols.append(parts[0])
            geometry.append([float(x) for x in parts[1:4]])
        # print(unit, charge, multiplicity, symbols, geometry)
        return unit, charge, multiplicity, symbols, pnp.array(geometry, requires_grad=False)

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


    def __init__(self, 
                 geometry, 
                 functional="PNOF4", 
                 conv_tol=1e-7, 
                 init_param=None, 
                 basis= 'sto-3g', 
                 max_iterations = 1000,
                 gradient="df_fedorov",
                 d_shift=1e-3):
        self.unit, self.charge, self.mul, self.symbols, self.crd = self._read_mol(geometry)
        self.basis = basis
        self.functional = functional
        self.conv_tol = conv_tol
        self.init_param = init_param
        self.max_iterations = max_iterations
        self.gradient = gradient
        self.d_shift = d_shift
        self.init_param_default = 0.1
        if init_param is not None:
            self.init_param = init_param
        else:
            self.init_param = self.init_param_default
        self.opt_param = None
        self.opt_rdm1 = None

    # ---------------- Ansatz ----------------
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

    # ---------- integrals at a geometry (MO basis) ----------
    def _mo_integrals(self, crds):
        """Return (E_nuc, h_MO, I_MO, n_electrons, norb) at given geometry (bohr)."""
        mol = qml.qchem.Molecule(symbols = self.symbols, 
                                 coordinates = crds, 
                                 charge = self.charge, 
                                 mult = self.mul, 
                                 basis_name = self.basis, 
                                 unit = self.unit)
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

    def ene_pnof4(self, params, crds, rdm1=None):
        # Functions based on 1-RDM (J. Chem. Theory Comput. 2025, 21, 5, 2402–2413) and taked from the following repository:
        # https://github.com/felipelewyee/NOF-VQE
        
        #E_nuc, h_MO, I_MO, n_elec, norb = self._mo_integrals(crds*1.8897259886) 
        E_nuc, h_MO, I_MO, n_elec, norb = self._mo_integrals(crds) 
        
        # Fermi level
        F = int(n_elec / 2)

        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)
        n, vecs = self._get_no_on(rdm1,norb)
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
    
        return E_nuc + E1 + E2, rdm1

    # =========================
    # Energy minimization
    # =========================

    def _vqe(self, E_fn, params, crds):
        opt = optax.sgd(learning_rate=0.1)
        opt_state = opt.init(params)
        
        # energy function that depends only on params (geometry fixed)
        E_single = lambda p: E_fn(p, crds)
        # evaluate once
        E0, rdm1_0 = E_single(params)

        E_history = [E0]
        rdm1_history = [rdm1_0]
        params_history = [params]
        
        for it in range(self.max_iterations):
        
            # gradient only w.r.t. params, so we take the first component (energy)
            grad_fn = lambda p: E_fn(p, crds)[0]
            gradient = jax.grad(grad_fn)(params)
        
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)

            E_val, rdm1_val = E_single(params)

            params_history.append(params)
            E_history.append(E_val)
            rdm1_history.append(rdm1_val)
        
            g_maxabs = jnp.max(jnp.abs(gradient))
        
            #print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
        
            if g_maxabs <= self.conv_tol:
                break

        return E_history, params_history, rdm1_history


    def ene_vqe(self):
        E_history, params_history, rdm1_history = self._vqe(
            self.ene_pnof4, self.init_param, self.crd
            )
        self.opt_param = params_history[-1]
        self.opt_rdm1 = rdm1_history[-1]
        return E_history[-1], params_history[-1], rdm1_history[-1]

    def _nuclear_gradient_fedorov(self, params, crds, rdm1_opt, d_shift):
        """
        Compute nuclear gradient using central finite differences
        following Fedorov et al. JCP 154, 164103 (2021).

        Args:
            params: optimized VQE parameters
            crds (array): nuclear coordinates (Bohr)
            h (float): finite-difference step in Bohr

        Returns:
            grad (array): nuclear gradient, same shape as crds
        """
        grad = pnp.zeros_like(crds)

        # loop over all atoms and Cartesian components
        for a in range(crds.shape[0]):
            for xyz in range(3):
                # displaced geometries (full copies)
                crds_plus = crds.copy()
                crds_minus = crds.copy()
                # displaced geometries
                crds_plus[a, xyz] = crds[a, xyz] + d_shift
                crds_minus[a, xyz] = crds[a, xyz] - d_shift

                E_plus, _ = self.ene_pnof4(params, crds_plus, rdm1=rdm1_opt)
                E_minus, _ = self.ene_pnof4(params, crds_minus, rdm1=rdm1_opt)

                grad[a, xyz] = (E_plus - E_minus) / (2 * d_shift)

        return grad

    def _nuclear_gradient(self, params, crds, d_shift, warm_start=True):
        """
        Compute nuclear gradient using central finite differences for
        nuclear coordinates.
        For each coordinate, re-optimize VQE at R+h and R-h (warm-started).

        Args:
            params: optimized VQE parameters
            crds (array): nuclear coordinates (Bohr)
            h (float): finite-difference step in Bohr

        Returns:
            grad (array): nuclear gradient, same shape as crds
        """
        grad = pnp.zeros_like(crds)
        params_p = params
        params_m = params

        # loop over all atoms and Cartesian components
        for a in range(crds.shape[0]):
            for xyz in range(3):
                # displaced geometries (full copies)
                crds_plus = crds.copy()
                crds_minus = crds.copy()
                # displaced geometries
                crds_plus[a, xyz] = crds[a, xyz] + d_shift
                crds_minus[a, xyz] = crds[a, xyz] - d_shift

                # Warm start from the reference optimal params (or the last one)
                p_start_plus = params_p if warm_start else self.init_param
                p_start_minus = params_m if warm_start else self.init_param

                E_plus, p_plus, _ = self._vqe(self.ene_pnof4, p_start_plus, crds_plus)
                E_minus, p_minus, _ = self._vqe(self.ene_pnof4, p_start_minus, crds_minus)

                # Optional: update warm-start for the next coordinate
                if warm_start:
                    params_p = p_plus[-1]  # carry forward the latest parameters
                    params_m = p_minus[-1]  # carry forward the latest parameters

                grad[a, xyz] = (E_plus[-1] - E_minus[-1]) / (2 * d_shift)

        return grad
    
    def grad(self):
        """The gradient is a post-processing calculation that depends on first computing the energy by VQE optimization"""
        if self.opt_param is None:
            print("Optimal parameter is None. Proceeding to compute the vqe energy")
            _,self.opt_param, self.opt_rdm1 = self.ene_vqe()
        if self.gradient == "df_fedorov":
            grad = self._nuclear_gradient_fedorov(
            self.opt_param, self.crd, self.opt_rdm1, self.d_shift
            )
        elif self.gradient == "df_normal":
            grad = self._nuclear_gradient(
            self.opt_param, self.crd, self.d_shift
            )
        else:
            raise SystemExit("The chosen gradient option is either incorrect or not implemented")
        return grad

 
# =========================
# Run the calculation
# =========================
if __name__ == "__main__":
    xyz_file = "h2_bohr.xyz"
    cal = NOFVQE(xyz_file, 
                 functional="PNOF4", 
                 conv_tol=1e-5, 
                 init_param=0.1, 
                 basis='sto-3g', 
                 max_iterations=500,
                 gradient="df_fedorov",
                 d_shift=1e-3)

    # Run VQE
    E_min, params_opt, rdm1_opt = cal.ene_vqe()
    print("Min Ene VQE and param:", E_min, params_opt)
    
    # Nuclear gradient at optimized geometry (finite differences by Fedorov)
    grad_fedorov = cal.grad()
    print("Nuclear gradient (Fedorov):\n", grad_fedorov)