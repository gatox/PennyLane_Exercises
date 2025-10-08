import time
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp

import optax
import psi4
psi4.core.be_quiet()
import pynof
from scipy.linalg import eigh
from scipy.optimize import minimize
import re
import os

#The necessary libraries to run on an IBM QC.
from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_cloud_sdk_core.api_exception import ApiException
from qiskit_ibm_runtime.accounts.exceptions import InvalidAccountError


jax.config.update("jax_enable_x64", True)

class NOFVQE:

    @staticmethod
    def _read_mol(inputdata):
        """
        Reads either a filepath (XYZ-like file) or a geometry string, 
        and builds a psi4.Molecule object.

        Returns:
            unit (str): "bohr"
            charge (int)
            multiplicity (int)
            symbols (list[str])
            geometry (pnp.array): shape (natoms, 3)
            xyz_str (str): full xyz string representation (for later use)
            mol (psi4.core.Molecule): Psi4 molecule object
        """
        # --- Read XYZ data ---
        if "\n" in inputdata or "units" in inputdata:
            lines = [line.strip() for line in inputdata.splitlines() if line.strip()]
        else:
            with open(inputdata, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

        # Build xyz string (Psi4 expects full string with header)
        xyz_str = "\n".join(lines)
        xyz_str = "symmetry c1\n" + xyz_str
        units = lines[0].split()[1]  # first line: "units bohr" or "units angstrom"
        charge, multiplicity = map(int, lines[1].split())

        symbols = []
        geometry = []
        for line in lines[2:]:
            parts = line.split()
            symbols.append(parts[0])
            geometry.append([float(x) for x in parts[1:4]])

        geometry = pnp.array(geometry, requires_grad=False)

        # Build psi4 molecule directly from string
        mol = psi4.geometry(xyz_str)

        return units, charge, multiplicity, symbols, geometry, mol

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
    
    
    @staticmethod
    def _func_indix(functional):
        index = re.findall(r'\d+',functional)
        return int(index[0])
        
    def __init__(self, 
                 geometry, 
                 functional="PNOF4", 
                 conv_tol=1e-5, 
                 init_param=None, 
                 basis= 'sto-3g', 
                 max_iterations = 1000,
                 gradient="analytics",
                 d_shift=1e-4,
                 C_MO=None,
                 dev="simulator",
                 n_shots=None,
                 optimization_level=None,
                 resilience_level=None,
                 ):
        self.units, self.charge, self.mul, self.symbols, self.crd, self.mol = self._read_mol(geometry)
        #self.crd = np.array([[self.mol.x(i), self.mol.y(i), self.mol.z(i)] for i in range(self.mol.natom())])
        self.basis = basis
        self.functional = functional
        self.ipnof = self._func_indix(functional)
        self.conv_tol = conv_tol
        self.init_param = init_param
        self.max_iterations = max_iterations
        self.gradient = gradient
        self.d_shift = d_shift
        self.init_param_default = 0.1
        self.p = pynof.param(self.mol,self.basis)
        self.p.ipnof = self.ipnof
        self.p.RI = True
        if init_param is not None:
            self.init_param = init_param
        else:
            self.init_param = self.init_param_default
        self.opt_param = None
        self.opt_rdm1 = None
        self.opt_n = None
        self.opt_vecs = None
        self.opt_cj12 = None
        self.opt_ck12 = None
        self.C_AO_MO = None
        self.H_ao = None 
        self.I_ao = None 
        self.b_mnl = None
        self.C = None
        if self.gradient == "analytics" and C_MO == "guest_C_MO":
            print("reading C_MO guest")
            file = "pynof_C.npy"
            if os.path.exists(file):
                self.C = pynof.read_C(self.p.title)
        self.dev = dev
        if self.dev != "simulator":
            self.n_shots = n_shots
            self.optimization_level = optimization_level
            self.resilience_level = resilience_level

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
    
    # ---------- integrals at a geometry (MO basis) from pennylane ----------
    def _mo_integrals_pennylane(self, crd):
        """Return (E_nuc, h_MO, I_MO, n_electrons, norb) at given geometry (bohr)."""
        mol = qml.qchem.Molecule(symbols = self.symbols, 
                                 coordinates = crd, 
                                 charge = self.charge, 
                                 mult = self.mul, 
                                 basis_name = self.basis, 
                                 unit = self.units)
        core, h_MO, I_MO = qml.qchem.electron_integrals(mol)()  # MO integrals
        E_nuc = core[0]
        n_elec = mol.n_electrons
        norb = int(h_MO.shape[0])
        return jnp.array(E_nuc), jnp.array(h_MO), jnp.array(I_MO), n_elec, norb
    
    def _read_C_MO(self, C,S_ao,p):
        if C is None:
            _, C = eigh(self.H_ao, S_ao)
        C = pynof.check_ortho(C,S_ao,p)
        np.save(p.title+"_C.npy",C)
        print(f"saving {p.title}C.npy")
        return C
    
    # ---------- integrals at a geometry (MO basis) from pynof ----------
    def _mo_integrals_pynof(self):
        mol_local = self.mol
        p = self.p
        # Compute integrals with PyNOF (from AO to MO)
        S_ao, _, _, self.H_ao, self.I_ao, self.b_mnl, _ = pynof.compute_integrals(p.wfn,mol_local,p)
        self.C_AO_MO = self._read_C_MO(self.C,S_ao,p)
        h_MO,I_or_b_MO = pynof.JKH_MO_tmp(self.C_AO_MO,self.H_ao,self.I_ao,self.b_mnl,p)
        if self.p.RI:
            b_MO = I_or_b_MO
            I_MO = np.einsum("pql,rsl->prsq", b_MO, b_MO, optimize=True)
        else:
            I_MO = np.transpose(I_or_b_MO, axes=(0,2,3,1))
        norb = int(h_MO.shape[0])
        E_nuc = mol_local.nuclear_repulsion_energy()
        n_elec = p.ne
        return jnp.array(E_nuc), jnp.array(h_MO), jnp.array(I_MO), n_elec, norb
    
    def _mo_integrals(self, crd):
        if self.gradient == "analytics":
            E_nuc, h_MO, I_MO, n_elec, norb = self._mo_integrals_pynof()
        else:
            E_nuc, h_MO, I_MO, n_elec, norb = self._mo_integrals_pennylane(crd)
        return E_nuc, h_MO, I_MO, n_elec, norb

    # ---------- measure 1-RDM on the circuit ----------
    def _rdm1_from_circuit(self, params, n_elec, norb, allow_fallback=False):
        max_retries = 10
        retry_delay = 5
        qubits = 2 * norb
        hf_state = [1] * n_elec + [0] * (qubits - n_elec)
        if self.dev == "simulator":
            dev = qml.device("lightning.qubit", wires=qubits)
        else:
            # Only initialize IBM service once
            if not hasattr(self, "_ibm_service"):
                self._ibm_service = QiskitRuntimeService()
            # Attempt IBM Q with retries
            for attempt in range(max_retries):
                try:
                    if self.dev == "noise_simulator":
                        # Load fake IBM backend and build noisy simulator
                        from qiskit_aer import AerSimulator
                        backend = self._ibm_service.backend("ibm_pittsburgh")
                        #AER Simulator#
                        backend = AerSimulator.from_backend(backend) 
                    elif self.dev == "real":
                        #backend = self._ibm_service.backend("ibm_pittsburgh")
                        backend = self._ibm_service.least_busy(operational=True, simulator=False)
                    backend.set_options(
                        max_parallel_threads = 0,
                        max_parallel_experiments = 0,
                        max_parallel_shots = 1,
                        statevector_parallel_threshold = 16,
                    )
                    # Use PennyLane device real/with noisy simulator
                    dev = qml.device('qiskit.remote', 
                                    wires=backend.num_qubits,
                                    backend=backend,
                                    optimization_level=self.optimization_level,
                                    resilience_level=self.resilience_level,
                                    shots=self.n_shots)
                    break  # success, exit retry loop
                except (ApiException, InvalidAccountError) as e:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IBM Q call failed (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
            else:
                if allow_fallback:
                    print("IBM Q unavailable after retries. Falling back to local AerSimulator.")
                    from qiskit_aer import AerSimulator
                    backend = AerSimulator()
                    dev = qml.device(
                        "qiskit.aer",
                        wires=backend.configuration().num_qubits,
                        backend=backend,
                        shots=self.n_shots,
                    )
                else:
                    raise RuntimeError("IBM Q unavailable after retries")
        @qml.qnode(dev, interface="jax")
        def rdm1_qnode(theta):
            self._ansatz(theta, hf_state, qubits)
            return [qml.expval(op) for op in self._build_rdm1_ops(norb)]

        return jnp.array(rdm1_qnode(params))

    def ene_pnof4(self, params, crds, rdm1=None):
        # Functions based on 1-RDM (J. Chem. Theory Comput. 2025, 21, 5, 2402–2413) and taked from the following repository:
        # https://github.com/felipelewyee/NOF-VQE
        
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
        
        #NEW: Compute cj12 and ck12 (like pynof.CJCKD4)
        n_outer = jnp.outer(n, n)
        cj12 = 2.0 * (n_outer - Delta)
        ck12 = n_outer - Delta - Pi
    
        return E_nuc + E1 + E2, rdm1, n, vecs, cj12, ck12

    # =========================
    # Energy minimization
    # =========================

    def _vqe_optax(self, E_fn, method, params, crds):
        # choose optimizer
        if method == "adam":
            opt = optax.adam(learning_rate=0.05)
        elif method == "sgd":
            opt = optax.sgd(learning_rate=0.1)
        else:
            raise ValueError(f"Unknown optax method: {method}")
        
        opt_state = opt.init(params)
        
        # energy function that depends only on params (geometry fixed)
        E_single = lambda p: E_fn(p, crds)
        # evaluate once
        E0, rdm1_0, n, vecs, cj12, ck12 = E_single(params)

        E_history = [E0]
        rdm1_history = [rdm1_0]
        params_history = [params]
        n_history = [n]
        vecs_history = [vecs]
        cj12_history = [cj12]
        ck12_history = [ck12]
        
        for it in range(self.max_iterations):
        
            # gradient only w.r.t. params, so we take the first component (energy)
            grad_fn = lambda p: E_fn(p, crds)[0]
            gradient = jax.grad(grad_fn)(params)
        
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)

            E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_single(params)

            params_history.append(params)
            E_history.append(E_val)
            rdm1_history.append(rdm1_val)
            n_history.append(n_val)
            vecs_history.append(vecs_val)
            cj12_history.append(cj12_val)
            ck12_history.append(ck12_val)
        
            g_maxabs = jnp.max(jnp.abs(gradient))
        
            #print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
        
            if g_maxabs <= self.conv_tol:
                break

        return E_history, params_history, rdm1_history, n_history, vecs_history, cj12_history, ck12_history


    def _vqe_scipy(self, E_fn, method, params, crds):
        # Define wrappers for scipy
        def E_scipy(x):
            E_val, _, _, _, _, _ = E_fn(x, crds)
            return float(E_val)

        def grad_scipy(x):
            grad_fn = lambda p: E_fn(p, crds)[0]
            g = jax.grad(grad_fn)(x)
            return np.array(g, dtype=float)

        res = minimize(
            E_scipy,
            np.array(params),
            method=method.upper(),
            jac=grad_scipy,
            tol=self.conv_tol,
            options={"maxiter": self.max_iterations},
        )

        E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(res.x, crds)
        return [E_val], [res.x], [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]


    def _vqe(self, E_fn, params, crds, method="adam"):
        if method.lower() in ["sgd", "adam"]:
            return self._vqe_optax(E_fn, method, params, crds)
        elif method.lower() in ["slsqp", "l-bfgs-b"]:
            return self._vqe_scipy(E_fn, method, params, crds)
        else:
            raise ValueError(
                f"Optimizer method {method} not implemented. Choose 'adam', 'sgd', 'slsqp', or 'l-bfgs-b'."
            )

    def ene_vqe(self):
        E_history, params_history, rdm1_history, n_history, vecs_history, cj12_history, ck12_history = self._vqe(
            self.ene_pnof4, self.init_param, self.crd
            )
        self.opt_param = params_history[-1]
        self.opt_rdm1 = rdm1_history[-1]
        self.opt_n = n_history[-1]
        self.opt_vecs = vecs_history[-1]
        self.opt_cj12 = cj12_history[-1]
        self.opt_ck12 = ck12_history[-1]
        return E_history[-1], params_history[-1], rdm1_history[-1], n_history[-1], vecs_history[-1], cj12_history[-1], ck12_history[-1]

    def _nuclear_gradient_dff_fedorov(self, params, crds, rdm1_opt, d_shift):
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

                E_plus, _, _, _, _, _ = self.ene_pnof4(params, crds_plus, rdm1=rdm1_opt)
                E_minus, _, _, _, _, _ = self.ene_pnof4(params, crds_minus, rdm1=rdm1_opt)

                grad[a, xyz] = (E_plus - E_minus) / (2 * d_shift)
        return grad

    def _nuclear_gradient_dff_normal(self, params, crds, d_shift, warm_start=True):
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

                E_plus, p_plus, _, _, _, _, _ = self._vqe(self.ene_pnof4, p_start_plus, crds_plus)
                E_minus, p_minus, _, _, _, _, _ = self._vqe(self.ene_pnof4, p_start_minus, crds_minus)

                # Optional: update warm-start for the next coordinate
                if warm_start:
                    params_p = p_plus[-1]  # carry forward the latest parameters
                    params_m = p_minus[-1]  # carry forward the latest parameters

                grad[a, xyz] = (E_plus[-1] - E_minus[-1]) / (2 * d_shift)

        return grad
    
    def _nuclear_gradient_analytics(self):
        """
        Computing the nuclear gradient using PyNOF package (from: 
        https://github.com/felipelewyee/PyNOF), inspired by 
        I. Mitxelena and M. Piris JCP 146, 014102 (2017)

        Args:
            the required parameters are computed once the VQE is called

        Returns:
            grad (array): nuclear gradient, same shape as crds
        """
        C_AO_MO = self.C_AO_MO
        V_MO_NO = np.array(self.opt_vecs)  # columns are NOs in MO basis
        # Build AO->NO coefficients
        C_AO_NO = np.dot(self.C_AO_MO, V_MO_NO)   # shape (nbf, norb) -> AO -> NO 
        C_AO_NO = np.array(C_AO_NO)       
        n_np = np.array(self.opt_n)
        cj12_np = np.array(self.opt_cj12)
        ck12_np = np.array(self.opt_ck12)
        H,I,b_mnl = self.H_ao, self.I_ao, self.b_mnl
        if(self.p.no1==0):
            elag,_ = pynof.computeLagrange2(n_np,cj12_np,ck12_np,C_AO_MO,H,I,b_mnl,self.p)
        else:
            J,K = pynof.computeJKj(C_AO_MO,I,b_mnl,self.p)
            if(p.MSpin==0):
                F = pynof.computeF_RC_driver(J,K,n_np,H,cj12_np,ck12_np,self.p)
            elif(not p.MSpin==0):
                F = pynof.computeF_RO_driver(J,K,n_np,H,cj12_np,ck12_np,self.p)
            elag = pynof.computeLagrange(F,C_AO_MO,self.p)
        return pynof.compute_geom_gradients(self.p.wfn,self.mol,n_np,C_AO_NO,cj12_np,ck12_np,elag,self.p)
    
    def grad(self):
        """The gradient is a post-processing calculation that depends on first computing the energy by VQE optimization"""
        if self.opt_param is None:
            print("Optimal parameter is None. Proceeding to compute the vqe energy")
            _,self.opt_param, self.opt_rdm1 = self.ene_vqe()
        if self.gradient == "df_fedorov":
            grad = self._nuclear_gradient_dff_fedorov(
            self.opt_param, self.crd, self.opt_rdm1, self.d_shift
            )
        elif self.gradient == "df_normal":
            grad = self._nuclear_gradient_dff_normal(
            self.opt_param, self.crd, self.d_shift
            )
        elif self.gradient == "analytics":
            grad = self._nuclear_gradient_analytics()
        else:
            raise SystemExit("The chosen gradient option is either incorrect or not implemented")
        return grad

 
# =========================
# Run the calculation
# =========================
if __name__ == "__main__":
    xyz_file = "h2_bohr.xyz"
    functional="PNOF4"
    conv_tol=1e-1
    init_param=0.1
    basis='sto-3g'
    max_iterations=500
    #gradient="df_fedorov"
    gradient="analytics"
    d_shift=1e-4
    C_MO = "guest_C_MO"
    dev="simulator"
    n_shots=10000
    optimization_level=3,
    resilience_level=0,
    cal = NOFVQE(
            xyz_file, 
            functional=functional, 
            conv_tol=conv_tol, 
            init_param=init_param, 
            basis=basis, 
            max_iterations=max_iterations,
            gradient=gradient,
            d_shift=d_shift,
            C_MO=C_MO,
            dev=dev,
            n_shots=n_shots,
            optimization_level=optimization_level,
            resilience_level=resilience_level,
                 )
    # Run VQE
    E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.ene_vqe()
    print("Min Ene VQE and param:", E_min, params_opt)
    # Nuclear gradient
    grad = cal.grad()
    print(f"Nuclear gradient ({gradient}):\n", grad)    
