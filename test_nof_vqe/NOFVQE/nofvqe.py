import time
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


from pennylane import FermiC, FermiA
from pennylane import jordan_wigner

import jax
from jax import numpy as jnp
from jax.scipy.linalg import expm

import optax
import psi4
psi4.core.be_quiet()
import pynof
from scipy.linalg import eigh
from scipy.optimize import minimize
import re
import os
import cma

#The necessary libraries to run on an IBM QC.
from qiskit_ibm_runtime import QiskitRuntimeService, Session
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
    def _get_no_on(rdm1, norb, pair_doubles):
        rdm1_aa = jnp.zeros((norb, norb))
        i = -1
        for p in range(norb):
            for q in range(p, norb):
                i += 1
                val = jnp.squeeze(rdm1[i])
                rdm1_aa = rdm1_aa.at[p, q].set(val)
                rdm1_aa = rdm1_aa.at[q, p].set(val)
        # If we enforce Ωg and only pair doubles → RDM1 already diagonal
        if pair_doubles:
            n = jnp.diag(rdm1_aa)
            vecs = jnp.eye(norb)
            return n, vecs

        # Otherwise, do the general case
        n, vecs = jnp.linalg.eigh(rdm1_aa)
        n = n[::-1]
        vecs = vecs[:, ::-1]
        print("ON_Edison:",n)
        print("NO_Edison:",vecs)
        return n, vecs
    
    @staticmethod
    def _func_indix(functional):
        index = re.findall(r'\d+',functional)
        return int(index[0])
        
    def __init__(self, 
                 geometry, 
                 functional="pnof4", 
                 conv_tol=1e-5, 
                 init_param=None, 
                 basis= 'sto-3g', 
                 max_iterations = 1000,
                 gradient="analytics",
                 pair_double=False,
                 d_shift=1e-4,
                 C_MO=None,
                 dev="simulator",
                 opt_circ="sgd",
                 n_shots=None,
                 optimization_level=None,
                 resilience_level=None,
                 ):
        self.units, self.charge, self.mul, self.symbols, self.crd, self.mol = self._read_mol(geometry)
        #self.crd = np.array([[self.mol.x(i), self.mol.y(i), self.mol.z(i)] for i in range(self.mol.natom())])
        self.basis = basis
        self.functional = functional
        self.ipnof = None
        if functional == "pnof4":
            self.ipnof = self._func_indix(functional)
        self.conv_tol = conv_tol
        self.max_iterations = max_iterations
        self.gradient = gradient
        self.d_shift = d_shift 
        self.p = pynof.param(self.mol,self.basis)
        self.p.ipnof = self.ipnof
        self.p.RI = False
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
        self.energy_scale = 1e3  # mHa
        self.icall = 0
        self.pair_doubles = pair_double
        if self.gradient == "analytics" and C_MO == "guest_C_MO":
            print("searching for C_MO guest")
            file_C = "pynof_C.npy"
            if os.path.exists(file_C):
                print("reading C_MO guest")
                self.C = pynof.read_C(self.p.title)
            else:
                print("No C_MO guest, then C_MO=None")
        self.dev = dev
        if self.dev != "simulator":
            self.n_shots = n_shots
            self.optimization_level = optimization_level
            self.resilience_level = resilience_level
        self.opt_circ = opt_circ
        if self.functional != "pnof4":
            self.pl_mol = qml.qchem.Molecule(
            symbols=self.symbols,
            coordinates=self.crd,
            unit = self.units
            )
            self.init_param = 0.1
        else:
            if self.pair_doubles:
                #self.init_param = init_param
                self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(self.crd)
                self.init_param = self._initial_params(init_param)
                self.n_params = len(self.init_param)
                self.n_kappa = int(self.n_elec/2)
            else:
                self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(self.crd)
                self.init_param = self._initial_params(init_param)
            
    # ---------------- Generate initial Parameters ----------------
    def _initial_params(self, params):
        if params is None:
            print("Params is none:", params)
            n_params = len(self.singles) + len(self.doubles)
            #params = np.random.normal(scale=0.05, size=n_params)
            #print("Params after random values:", params)
            params = np.full(n_params, 0.1)
            print("Params after fill with 0.1 value:", params)
        else:
            print("Params is not none:", params)
        return params


    # ---------------- Ansatz ----------------
    def _ansatz(self, params, hf_state, qubits):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.DoubleExcitation(params, wires=[0, 1, 2, 3])

    # ---------------- Ansatz 2----------------
    def _ansatz_2(self, params, hf_state, qubits):
        qml.BasisState(hf_state, wires=range(qubits))
        # apply all single excitations
        i = -1
        for s in self.singles:
            i = i + 1
            qml.SingleExcitation(params[i], wires=s)
        # apply all double excitations
        for d in self.doubles:
            i = i + 1
            qml.DoubleExcitation(params[i], wires=d)

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
    
    def _read_C_MO(self, C,S_ao,p):
        if self.C_AO_MO is None:
            if C is None:
                _, C = eigh(self.H_ao, S_ao)
            C_old = pynof.check_ortho(C,S_ao,p)
            for i in range(p.ndoc):
                for j in range(p.ncwo):
                    k = p.no1 + p.ndns + (p.ndoc - i - 1) * p.ncwo + j
                    l = p.no1 + p.ndns + (p.ndoc - i - 1) + j*p.ndoc
                    C[:,l] = C_old[:,k]
            if not self.pair_doubles:
                np.save(p.title+"_C.npy",C)
                print(f"saving {p.title}C.npy")
        else:
            C_old = self.C_AO_MO
        return C_old
    
    def _n_to_gamma_softmax(self, n):
        """
        Convert measured occupation numbers n_p into
        gamma parameters for PyNOF (Softmax scheme).
        """
        assert n.ndim == 1, f"Expected occupation vector, got shape {n.shape}"

        p = self.p

        # Convert ONCE to NumPy (safe: no gradients, no JIT)
        n_np = np.asarray(n, dtype=float)

        # Build reordered occupation vector expected by PyNOF
        n_reordered = np.copy(n_np)
        
        for i in range(p.ndoc):
            for j in range(p.ncwo):
                k = p.no1 + p.ndns + (p.ndoc - i - 1) * p.ncwo + j
                l = p.no1 + p.ndns + (p.ndoc - i - 1) + j*p.ndoc
                n_reordered[k] = n_np[l]
        p.nv = p.nbf5 - p.no1 - p.nsoc 

        assert np.all(n_reordered >= -1e-8)
        assert np.all(n_reordered <= 2.0 + 1e-8)

        # Map n -> gamma (PyNOF expects NumPy)  
        gamma = pynof.n_to_gammas_softmax(
            n_reordered,
            p.no1,
            p.ndoc,
            p.ndns,
            p.ncwo
        )
        return gamma
    
    def _orbital_optimization(self, gamma):
        """
        Perform classical orbital optimization using PyNOF.
        """
        p = self.p
        C = self.C_AO_MO
        H = self.H_ao
        I = self.I_ao
        b_mnl = self.b_mnl
        print("Printing MO old:",C)
        E_orb, C_new, nit, success = pynof.orbopt_adam(
            gamma, C, H, I, b_mnl, p
        )
        print("Printing MO new:",C_new)
        if not success:
            print("Warning: orbital optimization did not fully converge")

        # Save updated orbitals
        self.C_AO_MO = C_new
        
        np.save(p.title + "_C.npy", C_new)
        print(f"saving {p.title}C.npy")

        return E_orb

    def run_scnofvqe(self, max_outer=10, tol=1e-6):
        """
        Self-consistent NOF-VQE loop:
        VQE amplitudes <-> orbital optimization
        """
        gamma = None
        E_old = None
        n_old = None
        E_orb_old = None
        init_param = self.init_param
        for it in range(max_outer):
            print(f"\n==== SC-NOFVQE Iteration {it} ====")

            # 1. Build integrals with current orbitals
            self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(self.crd)
        
            self.init_param = self._initial_params(init_param)

            # 2. Run VQE (pair-only)
            E, params, rdm1, n, vecs, cj12, ck12 = self.ene_vqe()
            print("Updating initial parameters with optimal parameters",params)
            init_param = params
            print(f"VQE energy: {E:.10f} Ha")

            # # 3. Convergence check (NEW)
            # if n_old is not None:
            #     dn = np.linalg.norm(n - n_old)
            #     print(f"NO diff = {dn:.3e}")
            #     if abs(E - E_old) < tol and dn < tol:
            #         print("SC-NOFVQE converged (occupations)")
            #         break
            
            # E_old = E
            # n_old = n

            # 4. Convert occupations -> gamma
            gamma = self._n_to_gamma_softmax(n)

            # 5. Orbital optimization
            E_orb = self._orbital_optimization(gamma)
            print(f"Orbital optimization energy: {E_orb:.10f} Ha")

            # Convergence check (VQE and Orbital energy)
            if E_orb_old is not None:
                # if abs(E - E_old) < 1e-4 and abs(E_orb - E_orb_old) < 1e-4:
                if abs(E_orb - E_orb_old) < 1e-4:
                    print("SC-NOFVQE converged (occupations)")
                    break
            E_old = E
            #n_old = n
            E_orb_old = E_orb

        self.init_param = init_param
        return E, params, rdm1, n, vecs, cj12, ck12
    
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
    
    # ---------- integrals at a geometry (MO basis) from pynof ----------
    def _mo_integrals_pynof(self):
        mol_local = self.mol
        p = self.p
        # Compute integrals with PyNOF (from AO to MO)
        S_ao, _, _, self.H_ao, self.I_ao, self.b_mnl, _ = pynof.compute_integrals(p.wfn,mol_local,p)
        self.C_AO_MO = self._read_C_MO(self.C, S_ao,p)
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
            
        self.singles, self.doubles = qml.qchem.excitations(n_elec, 2 * norb)
        
        if self.pair_doubles:
            # Kill singles completely (seniority-zero ansatz)
            self.singles = []

            # Keep only pair doubles
            #self.doubles = self._filter_pair_doubles(self.doubles)
            Omega = self._build_pnof5e_omega(norb, n_elec)
            self.doubles = Omega 
        print("Size Singles:",len(self.singles))
        print("Singles:",self.singles)
        print("Size Doubles:",len(self.doubles))
        print("Doubles:",self.doubles)
        return E_nuc, h_MO, I_MO, n_elec, norb

    def _wrap_angles(self, p):
        p = np.asarray(p, dtype=float)
        # Map each parameter to (-pi, pi]
        return ((p + np.pi) % (2*np.pi)) - np.pi
    
    def _spin_orbital_pairs(self, start, n_pairs):
        """
        Build consecutive spin-orbital pairs:
        [[start, start+1], [start+2, start+3], ...]
        """
        return [[start + 2*i, start + 2*i + 1] for i in range(n_pairs)]
    
    def _omega_subspaces(self, virtual_pairs, F):
        """
        Split virtual MO pairs into Ω-subspaces of size F
        """
        return [
            virtual_pairs[i*F:(i+1)*F]
            for i in range(len(virtual_pairs) // F)
        ]

    def _build_pnof5e_omega(self, norb, n_elec):
        """
        Deterministic Ω_g construction for PNOF5e
        """
        F = n_elec // 2
        Nv = norb - F
        
        assert Nv % F == 0, "Virtual space not divisible by number of pairs"

        occ_pairs = self._spin_orbital_pairs(0, F)[::-1]

        virt_pairs = self._spin_orbital_pairs(2*F, norb - F)

        omegas = self._omega_subspaces(virt_pairs, F)

        # Build excitations
        doubles = []
        for i, occ in enumerate(occ_pairs):
            if i >= len(omegas):
                break
            for virt in omegas[i]:
                doubles.append(occ + virt)

        return doubles


    # ---------- measure 1-RDM on the circuit ----------
    #def _rdm1_from_circuit(self, params, n_elec, norb, region="eu-de", allow_fallback=False):
    def _rdm1_from_circuit(self, params, n_elec, norb, region="us-east", allow_fallback=False):
        max_retries = 10
        retry_delay = 5
        qubits = 2 * norb
        hf_state = [1] * n_elec + [0] * (qubits - n_elec)
        if self.dev in ["simulator", "hybrid"]:
            print("Calling simulator or hybrid:")
            # Hybrid mode uses simulator device for initial optimization
            dev = qml.device("lightning.qubit", wires=qubits)
        else:
            # Only initialize IBM service once
            if not hasattr(self, "_ibm_service"):
                if region == "eu-de":
                    #self._ibm_service = QiskitRuntimeService(name="eu-de-ibm-quantum-platform")
                    #self._ibm_service = QiskitRuntimeService(name="generic_eu-de-ibm-quantum-platform")  # generic_nofvqe
                    self._ibm_service = QiskitRuntimeService(name="solving_equation_eu-de-ibm-quantum-platform")  # solving_eq_nofvqe
                elif region == "us-east":
                    self._ibm_service = QiskitRuntimeService(name="us-east-ibm-quantum-platform")
                else:
                    raise RuntimeError(f"Unsupported region: {region}")
            # Attempt IBM Q with retries
            for attempt in range(max_retries):
                try:
                    if self.dev == "noise_simulator":
                        print("Calling noise_simulator:")
                        # Load fake IBM backend and build noisy simulator
                        from qiskit_aer import AerSimulator
                        backend = self._ibm_service.backend("ibm_pittsburgh")
                        #AER Simulator#
                        backend = AerSimulator.from_backend(backend) 
                        backend.set_options(
                        max_parallel_threads = 0,
                        max_parallel_experiments = 0,
                        max_parallel_shots = 1,
                        statevector_parallel_threshold = 16,
                        )
                    elif self.dev == "real":
                        print("Calling real QC:")
                        # Choose explicitly if region has only one backend
                        if region == "eu-de":
                            backend = self._ibm_service.backend("ibm_basquecountry")
                        else:
                            backend = self._ibm_service.backend("ibm_kingston")
                            #backend = self._ibm_service.least_busy(operational=True, simulator=False, system="heron2")
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
            #self._ansatz(theta, hf_state, qubits)
            self._ansatz_2(theta, hf_state, qubits)
            return [qml.expval(op) for op in self._build_rdm1_ops(norb)]

        # if self.pair_doubles:
        #     rdm1_ut = jnp.array(rdm1_qnode(params))  # length = norb*(norb+1)//2

        #     rdm1 = jnp.zeros((norb, norb))
        #     k = 0
        #     for p in range(norb):
        #         for q in range(p, norb):
        #             rdm1 = rdm1.at[p, q].set(rdm1_ut[k])
        #             rdm1 = rdm1.at[q, p].set(rdm1_ut[k])
        #             k += 1
        #     assert rdm1.ndim == 2 and rdm1.shape[0] == rdm1.shape[1], \
        #     f"RDM1 has invalid shape {rdm1.shape}"
        # else:
        #     params = jnp.atleast_1d(jnp.asarray(params))
        #     rdm1 = jnp.array(rdm1_qnode(params))
        #     # Flatten rdm1 if using SLSQP or L-BFGS-B
        #     if self.opt_circ in ["slsqp", "l-bfgs-b", "cobyla"]:
        #         rdm1 = rdm1.flatten()
        params = jnp.atleast_1d(jnp.asarray(params))
        rdm1 = jnp.array(rdm1_qnode(params))
        # Flatten rdm1 if using SLSQP or L-BFGS-B
        if self.opt_circ in ["slsqp", "l-bfgs-b", "cobyla"]:
            rdm1 = rdm1.flatten()
        return rdm1
    
    def ene_hf(self, params):
        H, qubits  = qml.qchem.molecular_hamiltonian(self.pl_mol)
        print("Number of qubits:", qubits)
        dev = qml.device("lightning.qubit", wires=qubits)
        @qml.qnode(dev, interface="jax")
        def hf_qnode(theta):
            qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
            qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
            return qml.expval(H)
        val = hf_qnode(params)       
        val = jnp.squeeze(val)     
        return val, None, None, None, None, None

    def ene_pnof(self, params, rdm1=None):
        # Functions based on 1-RDM (J. Chem. Theory Comput. 2025, 21, 5, 2402–2413) and taked from the following repository:
        # https://github.com/felipelewyee/NOF-VQE
        
        E_nuc, h_MO, I_MO, n_elec, norb = self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb
        # Fermi level
        F = int(n_elec / 2)

        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)
        n, vecs = self._get_no_on(rdm1,norb,self.pair_doubles)
        
        if self.pair_doubles:
            assert n.ndim == 1, f"Occupation numbers have wrong shape {n.shape}"
            assert vecs.ndim == 2, f"Orbital matrix has wrong shape {vecs.shape}"

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
    
    def build_pnof5_pair_array(self, Omega, norb):
        """
        Build fixed pair mapping from Ω_g structure.
        Each orbital is paired only with its Ω_g partners.
        """
        pair = -jnp.ones(norb, dtype=jnp.int64)

        for Omega_g in Omega:
            g0 = Omega_g[0]      # strong orbital
            for q in Omega_g[1:]:
                pair = pair.at[g0].set(q)
                pair = pair.at[q].set(g0)

        return pair

    
    def build_kappa_pairs(self, kappa_params, pair_of, norb):
        K = jnp.zeros((norb, norb))
        idx = 0

        for p in range(norb):
            for q in range(p+1, norb):

                # forbid intra-pair rotations
                if pair_of[p] == q:
                    continue

                if idx >= len(kappa_params):
                    break

                k = kappa_params[idx]
                idx += 1

                K = K.at[p, q].set(k)
                K = K.at[q, p].set(-k)

        return K



    def rotate_orbitals(self, vecs, kappa_params, pair_of):
        norb = vecs.shape[0]
        K = self.build_kappa_pairs(kappa_params, pair_of, norb)
        U = expm(K)
        return vecs @ U
    
    def build_pnof5e_omega_strong_weak(self, norb, n_elec):
        """
        Deterministic PNOF5e Ω_g construction.
        """
        F = n_elec // 2
        assert norb >= F

        strong = list(range(F))

        virtuals = list(range(F, norb))
        assert len(virtuals) % F == 0

        weak_g = [virtuals[i:i+F] for i in range(0,len(virtuals), F)]
        weak_g = weak_g[::-1]
        
        Omega = []
        
        for k,wg in zip(strong,weak_g):
            t = [k]
            t.extend(wg)
            Omega.append(t) 
        return Omega, strong, weak_g

    def ene_pnof4(self, x, rdm1=None):
        E_nuc, h_MO, I_MO, n_elec, norb = (
            self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb
        )

        if self.optimize_kappa:
            n_params = self.n_params
            params = x[:n_params]
            kappa_params = x[n_params:]
        else:
            params = x
            kappa_params = None

        F = n_elec // 2  # number of electron pairs (Fermi level)
        
        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)
            
        # Natural occupations and orbitals
        n, vecs = self._get_no_on(rdm1, norb, pair_doubles=True)
        
        # >>> ADD HERE <<<
        Omega, strong, weak = self.build_pnof5e_omega_strong_weak(norb, n_elec)
        pair_of = self.build_pnof5_pair_array(Omega, norb)
        # Rotation NO
        if kappa_params is not None:
            vecs = self.rotate_orbitals(vecs, kappa_params, pair_of)
        
        # Transform integrals to NO basis
        h_NO = jnp.einsum("ij,ip,jq->pq", h_MO, vecs, vecs, optimize=True)
        J_NO = jnp.einsum("ijkl,ip,jq,kq,lp->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)
        K_NO = jnp.einsum("ijkl,ip,jp,kq,lq->pq", I_MO, vecs, vecs, vecs, vecs, optimize=True)

        # --- Build CJ and CK (PNOF5) ---
        cj12 = 2.0 * jnp.outer(n, n)
        ck12 = jnp.outer(n, n)
        
        n_w = len(weak)
        # Pair structure (Ω_g)
        for g in range(F):
            p = g  # strongly occupied orbital
            q_start = F + g * n_w
            q_end   = q_start + n_w
            n_strong = n[p]
            n_weak = n[q_start:q_end]

            # Remove Coulomb intra-pair
            cj12 = cj12.at[p, q_start:q_end].set(0.0)
            cj12 = cj12.at[q_start:q_end, p].set(0.0)
            cj12 = cj12.at[q_start:q_end, q_start:q_end].set(0.0)

            # Exchange terms
            ck12 = ck12.at[p, q_start:q_end].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[q_start:q_end, p].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[q_start:q_end, q_start:q_end].set(
                -jnp.sqrt(jnp.outer(n_weak, n_weak))
            )
        
        # We have to change the way to pair them 
        # Omega_0 = []
        # for g in range(F):
        #     start = F + g * n_w
        #     end   = start + n_w
        #     Omega_g0 = [g] + list(range(start, end))
        #     Omega_0.append(Omega_g0)
        # print("EDISON_Omega_0:", Omega_0)
        
        Eg = 0.0
        E_pair = 0.0
        for g in range(F):
            for p in Omega[g]:
                Eg +=2.0 * n[p] * h_NO[p, p]
            Eg += n[g]*J_NO[g, g]
            for q in Omega[g][-F:]:
                Eg -= 2.0 *jnp.sqrt(n[g] * n[q]) * K_NO[q, g]
            for l in Omega[g][-F:]:
                for m in Omega[g][-F:]:
                    Eg += jnp.sqrt(n[m] * n[l]) * K_NO[l, m]
                    
        for j in range(F):
            for k in range(F):
                if j!=k:
                    for r in Omega[j]:
                        for s in Omega[k]:
                            E_pair += n[s] * n[r] *(2*J_NO[r, s]-K_NO[r, s])  
    
        return E_nuc + Eg + E_pair, rdm1, n, vecs, cj12, ck12

    # =========================
    # Circuit optimizers
    # =========================

    def _vqe_opt_pennylane(self, E_fn, method, params, max_iterations):
        """VQE optimization using SPSA"""
        # Choose optimizer
        # The selected hyperparameters are Pennylane's default.

        if method == "spsa":
            
            opt = qml.SPSAOptimizer(
                    maxiter=max_iterations, 
                    alpha=0.602, 
                    gamma=0.101, 
                    c=0.15, 
                    A=10, 
                    a=0.2
                    )
        else:
            raise ValueError(f"Unknown optax method: {method}")
        E_history = []
        params_history = []

        def cost_fn(p):
            E_val, _, _, _, _, _ = E_fn(p)
            return E_val
        
        for n in range(max_iterations):
            params, E_val = opt.step_and_cost(cost_fn, params)
            E_history.append(E_val)
            params_history.append(params)

            print(f"SPSA Step {n}: Energy = {E_val:.8f}")

            if n > 0 and abs(E_history[-1] - E_history[-2]) < self.conv_tol:
                break

        # Evaluate final rdm1 and other quantities
        _, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(params)
        return E_history, params_history, [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]


    def _vqe_optax(self, E_fn, method, params, max_iterations):
        """
        VQE optimization using:
            SGD and ADAM
        """

        # choose optimizer
        if method == "adam":
            opt = optax.adam(learning_rate=0.05)
        elif method == "sgd":
            opt = optax.sgd(learning_rate=0.1)
        else:
            raise ValueError(f"Unknown optax method: {method}")
        
        opt_state = opt.init(params)


        # scalar energy function for JAX
        def energy_only(p):
            E = E_fn(p)[0]
            return jnp.asarray(E).reshape(())
        
        # evaluate once
        E0, rdm1_0, n, vecs, cj12, ck12 = E_fn(params)

        E_history = [E0]
        rdm1_history = [rdm1_0]
        params_history = [params]
        n_history = [n]
        vecs_history = [vecs]
        cj12_history = [cj12]
        ck12_history = [ck12]
        
        for it in range(max_iterations):

            gradient = jax.grad(energy_only)(params)
        
            updates, opt_state = opt.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)

            E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(params)

            params_history.append(params)
            E_history.append(E_val)
            rdm1_history.append(rdm1_val)
            n_history.append(n_val)
            vecs_history.append(vecs_val)
            cj12_history.append(cj12_val)
            ck12_history.append(ck12_val)
        
            g_maxabs = jnp.max(jnp.abs(gradient))
        
            print(f"Step = {it},  Energy = {E_history[-1]:.8f} Ha,  Gradient = {g_maxabs:.1e}")
        
            if g_maxabs <= self.conv_tol:
                break

        return E_history, params_history, rdm1_history, n_history, vecs_history, cj12_history, ck12_history
    
    # Helper scipy minimizer
    def _run_scipy_minimize(self, E_fn, method, params, max_iterations):
        """
        VQE optimization using:
            SLSQP, L-BFGS-B and COBYLA
        """
        def E_scipy(x):
            x = jnp.array(x)
            E_val, *_ = E_fn(x)
            return float(E_val * self.energy_scale)

        use_analytic_jac = (
            method.lower() in ["slsqp", "l-bfgs-b"]
            and getattr(self, "dev", "simulator") == "simulator"
        )

        if use_analytic_jac:
            def grad_scipy(x):
                x = jnp.array(x)
                return np.array(jax.grad(lambda p: E_fn(p)[0])(x), dtype=float)
            jac = grad_scipy
        else:
            jac = None

        bounds = [(-np.pi, np.pi) for _ in range(len(np.atleast_1d(params)))]

        res = minimize(
            E_scipy,
            np.array(params, dtype=float),
            method=method.upper(),
            jac=jac,
            bounds=bounds,
            tol=self.conv_tol,
            options={
                "maxiter": max_iterations,
                "ftol": self.conv_tol,
                "eps": 1e-8,
            },
        )

        return res
    
    def _vqe_opt_scipy(self, E_fn, method, params, max_iterations):

        # -------------------------------------------------
        # STAGE 1 — θ only (NO discovery)
        # -------------------------------------------------
        self.optimize_kappa = False

        res_theta = self._run_scipy_minimize(
            E_fn,
            method,
            params,
            max_iterations,
        )

        theta_opt = self._wrap_angles(res_theta.x)

        # Evaluate once to store NOs etc.
        E1, rdm1, n, vecs, cj12, ck12 = E_fn(jnp.array(theta_opt))

        # -------------------------------------------------
        # Decide whether we stop here
        # -------------------------------------------------
        if not getattr(self, "orbital_optimization", True):
            return (
                [E1],
                [theta_opt],
                [rdm1],
                [n],
                [vecs],
                [cj12],
                [ck12],
            )

        # -------------------------------------------------
        # STAGE 2 — θ + κ (orbital optimization)
        # -------------------------------------------------
        self.optimize_kappa = False

        n_kappa = self.n_kappa  # you already define this
        kappa0 = np.zeros(n_kappa)

        x0 = np.concatenate([theta_opt, kappa0])

        res_full = self._run_scipy_minimize(
            E_fn,
            method,
            x0,
            max_iterations,
        )

        x_opt = self._wrap_angles(res_full.x)

        E2, rdm1, n, vecs, cj12, ck12 = E_fn(jnp.array(x_opt))

        return (
            [E2],
            [x_opt],
            [rdm1],
            [n],
            [vecs],
            [cj12],
            [ck12],
        )


    # def _vqe_opt_scipy(self, E_fn, method, params, max_iterations):
    #     """
    #     VQE optimization using:
    #         SLSQP, L-BFGS-B and COBYLA
    #     """
    #     # Define wrappers for scipy
    #     def E_scipy(x):
    #         x = jnp.array(x)  # ensure JAX array
    #         E_val, _, _, _, _, _ = E_fn(x)
    #         return float(E_val* self.energy_scale)
        
    #     # If running on a real/remote backend, do NOT supply an analytic jacobian to SciPy.
    #     # SciPy will use finite differences (function evaluations only) which is robust.
    #     use_analytic_jac = (method.lower() in ["slsqp", "l-bfgs-b"]) and (getattr(self, "dev", "simulator") == "simulator")
        
    #     #COBYLA optimizer doesn't support jacobian
    #     if use_analytic_jac:
    #         def grad_scipy(x):
    #             x = jnp.array(x)  # ensure JAX array
    #             grad_fn = lambda p: E_fn(p)[0]
    #             # This will call jax.grad - only safe on local simulator
    #             g = jax.grad(grad_fn)(x)
    #             return np.array(g, dtype=float)
    #         jac = grad_scipy
    #     else:
    #         # For remote devices or other methods, let SciPy approximate the jac via FD
    #         jac = None  # COBYLA doesn't use gradients

    #     bounds = [(-np.pi, np.pi) for _ in range(len(np.atleast_1d(params)))]

    #     iter_counter = {"i": 0}
    #     # this function prints the values per iterations, however, 
    #     # it computes the 1rdm two times more, so it is not recommended 
    #     # to use it in a real QC
    #     def callback(xk):
    #         x = jnp.array(xk)
    #         E_val, *_ = E_fn(x)

    #         if jac is not None:
    #             g = jac(xk)
    #             g_maxabs = np.max(np.abs(g))
    #             print(
    #                 f"Step = {iter_counter['i']}, "
    #                 f"Energy = {float(E_val):.8f} Ha, "
    #                 f"Gradient = {g_maxabs:.1e}"
    #             )
    #         else:
    #             print(
    #                 f"Step = {iter_counter['i']}, "
    #                 f"Energy = {float(E_val):.8f} Ha"
    #             )

    #         iter_counter["i"] += 1
    #     res = minimize(
    #         E_scipy,
    #         np.array(params, dtype=float),
    #         method=method.upper(),
    #         jac=jac,
    #         bounds=bounds,
    #         tol=self.conv_tol,
    #         options={
    #             "maxiter": max_iterations,
    #             "ftol": self.conv_tol,   # keep it
    #             "eps": 1e-8,             # critical
    #         },
    #         #callback=callback,
    #     )
        
    #     res_x = np.asarray(res.x, dtype=float)
    #     res_x_wrapped = self._wrap_angles(res_x)

    #     E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(jnp.array(res_x_wrapped))
    #     return [E_val], [res.x], [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]
    
    def _vqe_cmaes(self, E_fn, params, max_iterations):
        """VQE optimization using CMA-ES."""
        
        # --- Ensure params are 1D ---
        params = np.atleast_1d(np.array(params, dtype=float)).flatten()
        
        # Define cost function (energy only)
        def cost_fn(p):
            E_val, _, _, _, _, _ = E_fn(np.array(p))
            return float(E_val)
        
        # Initial standard deviation (step size)
        sigma0 = 0.1  
        
        # Initialize CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(np.array(params), sigma0, {'maxiter': max_iterations})
        
        E_history = []
        params_history = []
        
        while not es.stop():
            solutions = es.ask()
            energies = [cost_fn(s) for s in solutions]
            es.tell(solutions, energies)
            es.disp()
            
            # Track best
            best_energy = min(energies)
            best_params = solutions[np.argmin(energies)]
            E_history.append(best_energy)
            params_history.append(best_params)
            
            # Convergence check
            if len(E_history) > 1 and abs(E_history[-1] - E_history[-2]) < self.conv_tol:
                break
        
        # Final evaluation
        _, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(best_params)
        return E_history, params_history, [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]
    
    # =========================
    # Energy minimization
    # =========================

    def _vqe(self, E_fn, params, method=None, max_iterations=None):
        if method is None:
            method=self.opt_circ
        if max_iterations is None:
            max_iterations = self.max_iterations
        if method.lower() in ["sgd", "adam"]:
            return self._vqe_optax(E_fn, method, params, max_iterations)
        elif method.lower() in ["slsqp", "l-bfgs-b", "cobyla"]:
            return self._vqe_opt_scipy(E_fn, method, params, max_iterations)
        elif method == "spsa":
            return self._vqe_opt_pennylane(E_fn, method, params, max_iterations)
        elif method == "cmaes":
            return self._vqe_cmaes(E_fn, params, max_iterations)
        else:
            raise ValueError(
                f"Optimizer method {method} not implemented. Choose: 'adam', 'sgd', 'spsa', 'sgd', 'slsqp', or 'l-bfgs-b'."
            )

    def ene_vqe(self):
        if self.functional == "vqe":
            method_opt = "slsqp"
            print("==== HF_VQE ====")
            res = self._vqe_opt_scipy(self.ene_hf, method_opt, self.init_param, self.max_iterations)

            # res can be a tuple/list of length >= 2:
            # ([E_history], [params_history], [rdm1_history], [n_history], [vecs_history], ...)
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                E_history = res[0]
                params_history = res[1]
            else:
                # If the optimizer unexpectedly returned a single value, try to handle it
                raise ValueError("Unexpected return from _vqe_opt_scipy: need at least E_history and params_history")

            # Ensure we return the final (scalar) energy and final parameters
            E_final = E_history[-1] if len(E_history) > 0 else E_history
            params_final = params_history[-1] if len(params_history) > 0 else params_history
            self.opt_param = params_final
            return E_final, params_final, None, None, None, None, None
        elif self.functional == "pnof4":
            E_history, params_history, rdm1_history, n_history, vecs_history, cj12_history, ck12_history = self._vqe(
                self.ene_pnof4, self.init_param
                )
            self.opt_param = params_history[-1]
            self.opt_rdm1 = rdm1_history[-1]
            self.opt_n = n_history[-1]
            self.opt_vecs = vecs_history[-1]
            self.opt_cj12 = cj12_history[-1]
            self.opt_ck12 = ck12_history[-1]
            print("==== Hybrid mode deactivated ====")
            print("Device: ",str(self.dev))
            print("Opt_circ: ",self.opt_circ)
            print("===============================")
            if self.dev != "hybrid":
                return E_history[-1], params_history[-1], rdm1_history[-1], n_history[-1], vecs_history[-1], cj12_history[-1], ck12_history[-1]
            else:
                self.dev_old = self.dev
                #self.opt_circ_old = self.opt_circ
                self.dev = "real"
                #self.opt_circ = "adam"
                """ 
                Optimized values are first computed with a simulator. 
                Then, they are recalculated using a real QC.
                """
                # E_hybrid, params_hybrid, rdm1_hybrid, n_hybrid, vecs_hybrid, cj12_hybrid, ck12_hybrid = self._vqe(
                #     self.ene_pnof4, self.opt_param, self.crd, max_iterations=1)
                E_hybrid, rdm1_hybrid, n_hybrid, vecs_hybrid, cj12_hybrid, ck12_hybrid = self.ene_pnof4(
                    self.opt_param)
                print("==== Hybrid mode activated ====")
                print("Devise: ",str(self.dev))
                print("Opt_circ: ",self.opt_circ)
                print("===============================")
                self.dev = self.dev_old
                #self.opt_circ = self.opt_circ_old
                #self.opt_param = params_hybrid[-1]
                # self.opt_rdm1 = rdm1_hybrid[-1]
                # self.opt_n = n_hybrid[-1]
                # self.opt_vecs = vecs_hybrid[-1]
                # self.opt_cj12 = cj12_hybrid[-1]
                # self.opt_ck12 = ck12_hybrid[-1]

                #######The energy is ecaluated into the QC#######
                self.opt_rdm1 = rdm1_hybrid
                self.opt_n = n_hybrid
                self.opt_vecs = vecs_hybrid
                self.opt_cj12 = cj12_hybrid
                self.opt_ck12 = ck12_hybrid
                #################################################

                ## Diagnostic: how noise shifted the parameters
                ##delta_theta = params_hybrid[-1] - params_history[-1]
                ##print("(hardware - simulator) =", delta_theta)
                ##print("square norm =", np.linalg.norm(delta_theta))
                #return E_hybrid[-1], params_hybrid[-1], rdm1_hybrid[-1], n_hybrid[-1], vecs_hybrid[-1], cj12_hybrid[-1], ck12_hybrid[-1]
                return E_hybrid, params_history[-1], rdm1_hybrid, n_hybrid, vecs_hybrid, cj12_hybrid, ck12_hybrid
        else:
            raise ValueError(f"Unknown/unimplemented functional or method : {self.functional}")

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

                if self.functional != "pnof4":
                    self.pl_mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=crds_plus,
                        unit = self.units
                        )
                    E_plus, _, _, _, _, _ = self.ene_hf(params)
                    self.pl_mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=crds_minus,
                        unit = self.units
                        )
                    E_minus, _, _, _, _, _ = self.ene_hf(params)
                else:
                    self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(crds_plus)
                    E_plus, _, _, _, _, _ = self.ene_pnof4(params, rdm1=rdm1_opt)
                    self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(crds_minus)
                    E_minus, _, _, _, _, _ = self.ene_pnof4(params, rdm1=rdm1_opt)
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

                if self.functional != "pnof4":
                    self.pl_mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=crds_plus,
                        unit = self.units
                        )
                    E_plus, _, _, _, _, _ = self.ene_hf(params)
                    self.pl_mol = qml.qchem.Molecule(
                        symbols=self.symbols,
                        coordinates=crds_minus,
                        unit = self.units
                        )
                    E_minus, _, _, _, _, _ = self.ene_hf(params)
                else:
                    self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(crds_plus)
                    E_plus, p_plus, _, _, _, _, _ = self._vqe(self.ene_pnof4, p_start_plus)
                    self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(crds_minus)
                    E_minus, p_minus, _, _, _, _, _ = self._vqe(self.ene_pnof4, p_start_minus)

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
        if self.opt_param is None and self.gradient in ["df_fedorov", "df_normal"]:
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
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nofvqe.py <file.xyz>")
        sys.exit(1)

    xyz_file = sys.argv[1]
    functional="pnof4"
    #functional="vqe"
    conv_tol=1e-7
    #init_param=0.1
    init_param=None
    basis='sto-3g'
    max_iterations=500
    #gradient="analytics"
    gradient="df_fedorov"
    d_shift=1e-4
    C_MO = "guest_C_MO"
    dev="simulator"
    #opt_circ="sgd"
    opt_circ="slsqp"
    n_shots=10000
    optimization_level=3
    resilience_level=0
    pair_double = True
    cal = NOFVQE(
            xyz_file, 
            functional=functional, 
            conv_tol=conv_tol, 
            init_param=init_param, 
            basis=basis, 
            max_iterations=max_iterations,
            opt_circ=opt_circ,
            gradient=gradient,
            pair_double=pair_double,
            d_shift=d_shift,
            C_MO=C_MO,
            dev=dev,
            n_shots=n_shots,
            optimization_level=optimization_level,
            resilience_level=resilience_level,
                 )
    if pair_double:
        #E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.run_scnofvqe()
        E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.ene_vqe()
        print("Min Ene VQE and param:", E_min, params_opt)
        print("RDM1_opt:",rdm1_opt)
        print("ON:",n)
        print("NO:",vecs)
        
    else:
        E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.ene_vqe()
        print("Min Ene VQE and param:", E_min, params_opt)
        # Nuclear gradient
        grad = cal.grad()
        print(f"Nuclear gradient ({gradient}):\n", grad)
        
    # Run VQE
    # E_h, params_h, rdm1_h, n_h, vecs_h, cj12_h, ck12_h=cal._vqe(cal.ene_pnof4, init_param, crds, method="adam")
    # Run NOF-VQE
    # E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.ene_vqe()
    # print("Min Ene VQE and param:", E_min, params_opt)
    # print("params_opt:", params_opt)
    # print("rdm1_opt:", rdm1_opt)
    # print("n_opt:", n)
    # print("vecs_opt:", vecs)
    # # Nuclear gradient
    # grad = cal.grad()
    # print(f"Nuclear gradient ({gradient}):\n", grad)  
    # print("Crd:",cal.crd)
    # E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12 = cal.run_scnofvqe()
    # print("Min Ene VQE and param:", E_min, params_opt)