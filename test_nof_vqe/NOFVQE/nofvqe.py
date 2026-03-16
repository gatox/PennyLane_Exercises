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
from scipy.linalg import eigh, expm
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

        geom_jax = jnp.array(geometry)
        geometry = pnp.array(geometry, requires_grad=False)

        

        return units, charge, multiplicity, symbols, geometry, xyz_str, geom_jax
    
    @staticmethod
    def _get_no_on(rdm1, norb, pair_doubles, tol=1e-8):

        if pair_doubles:
            # Ensure Hermiticity (cheap + safe)
            rdm1 = 0.5 * (rdm1 + rdm1.T)

            # Check if RDM1 is already diagonal
            offdiag_norm = jnp.linalg.norm(
                rdm1 - jnp.diag(jnp.diag(rdm1))
            )

            if offdiag_norm < tol:
                #print("The RDM1 is already diagonal")
                # Ideal seniority-zero case
                n = jnp.diag(rdm1)
                vecs = jnp.eye(norb)

            else:
                #print("The RDM1 is not diagonal")
                # Noise / leakage case: diagonalize
                n_raw, vecs_raw = jnp.linalg.eigh(rdm1)

                # Sort occupations descending
                idx = jnp.argsort(n_raw)[::-1]
                n = n_raw[idx]
                vecs = vecs_raw[:, idx]

                # (Optional but recommended)
                # Enforce electron count conservation
                ne_pairs = jnp.trace(rdm1)
                n = n * (ne_pairs / jnp.sum(n))

            return n, vecs

        else:
            # ---- general (non-pair) case: unchanged ----
            rdm1_aa = jnp.zeros((norb, norb))
            i = -1
            for p in range(norb):
                for q in range(p, norb):
                    i += 1
                    val = jnp.squeeze(rdm1[i])
                    rdm1_aa = rdm1_aa.at[p, q].set(val)
                    rdm1_aa = rdm1_aa.at[q, p].set(val)

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
        self.units, self.charge, self.mul, self.symbols, self.crd, self.xyz_str, self.geom_jax = self._read_mol(geometry)
        self.basis = basis
        self.functional = functional
        self.natoms = len(self.symbols)
        self.ipnof = None
        if functional in ["pnof4","pnof5","pnof7","pnof8"]:
            self.ipnof = self._func_indix(functional)
        self.conv_tol = conv_tol
        self.max_iterations = max_iterations
        self.gradient = gradient
        self.d_shift = d_shift
        self.opt_param = None
        self.opt_rdm1 = None
        self.opt_n = None
        self.opt_vecs = None
        self.opt_cj12 = None
        self.opt_ck12 = None
        self.C_MO = C_MO
        self.H_ao = None 
        self.I_ao = None 
        self.b_mnl = None
        self.C = None
        self.maxloop = 30
        self.energy_scale = 1e3  # mHa
        self.pair_doubles = pair_double
        self.dev = dev
        if self.dev != "simulator":
            self.n_shots = n_shots
            self.optimization_level = optimization_level
            self.resilience_level = resilience_level
        self.opt_circ = opt_circ
        if self.functional not in ["pnof4","pnof5","pnof7","pnof8"]:
            self.pl_mol = qml.qchem.Molecule(
            symbols=self.symbols,
            coordinates=self.crd,
            unit = self.units
            )
            self.init_param = 0.1
        else:
            self.init_param = init_param
                
    # def _orthonormalize(self, C,S):
    #     eigval,eigvec = eigh(S) 
    #     S_12 = jnp.einsum('ij,j->ij',eigvec,eigval**(-1/2),optimize=True)

    #     Cnew = jnp.einsum('ik,kj->ij',S,C,optimize=True)

    #     Cnew2 = jnp.einsum('ki,kj->ij',S_12,Cnew)

    #     for i in range(Cnew2.shape[1]):
    #         norm = jnp.einsum('k,k->',Cnew2[:,i],Cnew2[:,i],optimize=True)
    #         Cnew2[:,i] = Cnew2[:,i]/jnp.sqrt(norm)
    #         for j in range(i+1,Cnew2.shape[1]):
    #             val = -jnp.einsum('k,k->',Cnew2[:,i],Cnew2[:,j],optimize=True)
    #             Cnew2[:,j] = Cnew2[:,j] + val*Cnew2[:,i]

    #     C = jnp.einsum("ik,kj->ij",S_12,Cnew2,optimize=True)

    #     return C
    
    def _orthonormalize(self, C, S):

        # overlap in MO basis
        M = C.T @ S @ C

        # eigen decomposition
        eigval, eigvec = eigh(M)

        # inverse sqrt
        M_inv_sqrt = eigvec @ jnp.diag(eigval**(-0.5)) @ eigvec.T

        # orthonormalized orbitals
        C_new = C @ M_inv_sqrt

        return C_new
        
                
    # def _check_ortho(self, C,S):
    #     # Revisa ortonormalidad
    #     orthonormality = True
    #     CTSC = np.matmul(np.matmul(np.transpose(C),S),C)
    #     ortho_deviation = np.abs(CTSC - np.identity(self.nbf))
    #     if (np.any(ortho_deviation > 10**-6)):
    #         orthonormality = False
    #     if not orthonormality:
    #         print("Orthonormality violations {:d}, Maximum Violation {:f}".format((ortho_deviation > 10**-6).sum(),ortho_deviation.max()))
    #         print("Trying to orthonormalize")
    #         C = self._orthonormalize(C,S)
    #         C = self._check_ortho(C,S)
    #     else:
    #         print("No violations of the orthonormality")
    #     for j in range(self.nbf):
    #         #Obtiene el índice del coeficiente con mayor valor absoluto del MO
    #         idxmaxabsval = 0
    #         for i in range(self.nbf):
    #             if(abs(C[i][j])>abs(C[idxmaxabsval][j])):
    #                 idxmaxabsval = i
    #     # Ajusta el signo del MO
    #     sign = np.sign(C[idxmaxabsval][j])
    #     C[0:self.nbf,j] = sign*C[0:self.nbf,j]

    #     return C
    
    def _check_ortho(self, C, S):

        CTSC = C.T @ S @ C
        ortho_deviation = jnp.abs(CTSC - jnp.eye(self.nbf))

        violation_mask = ortho_deviation > 1e-6

        if jnp.any(violation_mask):

            print(
                "Orthonormality violations {:d}, Maximum Violation {:f}".format(
                    violation_mask.sum(), ortho_deviation.max()
                )
            )
            print("Trying to orthonormalize")

            C = self._orthonormalize(C, S)
            C = self._check_ortho(C, S)          
            # # re-check
            # CTSC = C.T @ S @ C
            # ortho_deviation = jnp.abs(CTSC - jnp.eye(self.nbf))

        else:
            print("No violations of the orthonormality")

        # ----------------------------
        # Sign convention correction
        # ----------------------------

        # index of largest |C| in each MO column
        idxmax = jnp.argmax(jnp.abs(C), axis=0)

        # values at those indices
        col_indices = jnp.arange(self.nbf)
        max_vals = C[idxmax, col_indices]

        # sign of largest coefficient
        signs = jnp.sign(max_vals)

        # broadcast sign correction
        C = C * signs

        return C
    
    def _molecule_pnl(self, crd):
        mol = qml.qchem.Molecule(symbols = self.symbols, 
                                 coordinates = crd, 
                                 charge = self.charge, 
                                 mult = self.mul, 
                                 basis_name = self.basis, 
                                 unit = self.units)
        return mol
    
    def _ao_integrals_pnl(self, crd):
        """Compute S, T, V, H and ERIs (I) in atomic orbitals from Pennylane"""
        
        mol = self._molecule_pnl(crd)
        charges = jnp.asarray(mol.nuclear_charges)
        basis = mol.basis_set
        S = qml.qchem.overlap_matrix(basis)()
        T = qml.qchem.kinetic_matrix(basis)()
        V = qml.qchem.attraction_matrix(basis, charges, crd)()
        Hcore = T + V

        ERI = qml.qchem.repulsion_tensor(basis)()

        Enuc = jnp.squeeze(qml.qchem.nuclear_energy(charges, crd)())
        
        E_HF = qml.qchem.hf_energy(mol)()
        
        return S, Hcore, ERI, Enuc, mol, E_HF
    
    def _ao_integrals_psi4(self):
        """Compute S, T, V, H and ERIs (I) in atomic orbitals from Psi4"""
        
        # Build psi4 molecule directly from string
        self.mol = mol = psi4.geometry(self.xyz_str)
        psi4.set_options({'basis': self.basis})
        self.wfn = wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
        # Integrador
        mints = psi4.core.MintsHelper(self.wfn.basisset())

        # Overlap, Kinetics, Potential
        S = pnp.asarray(mints.ao_overlap(), requires_grad=True)
        T = pnp.asarray(mints.ao_kinetic(), requires_grad=True)
        V = jnp.asarray(mints.ao_potential())
        Hcore = T + V
        
        ERI = pnp.asarray(mints.ao_eri(), requires_grad=True)
        
        Enuc = mol.nuclear_repulsion_energy()
        E_HF, _ = psi4.energy("HF", return_wfn=True)
        
        return S, Hcore, ERI, Enuc, wfn, E_HF

    def _ao_integrals(self, crd):
        if self.gradient == "analytics":
            S,H,I,Enuc,wfn_mol,E_HF = self._ao_integrals_psi4()
        else:
            S,H,I,Enuc,wfn_mol,E_HF = self._ao_integrals_pnl(crd)
        return S,H,I,Enuc,wfn_mol,E_HF
        
    def _global_parameters(self, wfn_mol):
        
        if self.gradient == "analytics":
            wfn = wfn_mol
            # ---- Global parameters for Psi4----
            self.nalpha = wfn.nalpha()
            self.nbeta = wfn.nbeta()
            self.n_elec = self.nalpha + self.nbeta
            self.norb = wfn.nmo()
            
        else:
            # ---- Global parameters for Pennylane----
            mol = wfn_mol
            self.n_elec = mol.n_electrons
            self.norb = mol.n_orbitals
            S_mult = (self.mul - 1) / 2
            self.nalpha = int(self.n_elec//2 + S_mult)
            self.nbeta = int(self.n_elec//2 - S_mult)
        
        self.nbf = self.norb
        self.no1 = 0
        
        # ---- Fortran NBF < NE condition (Taken from DoNOF)----
        if self.nbf < self.n_elec:
            ndif = self.n_elec - self.nbf
            if self.no1 < ndif:
                self.no1 = ndif
                print("WARNING: NO1 increased to satisfy NBF >= NE condition")
            if self.no1 == self.nbf:
                raise RuntimeError("All orbitals excluded (NO1 == NBF).")
        # ------------------------------------
        
        self.ndoc = self.nbeta   -   self.no1
        self.nsoc = self.nalpha  -   self.nbeta
        self.ndns = self.ndoc    +   self.nsoc
        self.nvir = self.nbf     -   self.nalpha
        self.ncwo = -1
        if(self.n_elec==2):
            self.ncwo= -1
        if(self.ndns!=0):
            if(self.ndoc>0):
                if(self.ncwo!=1):
                    if(self.ncwo==-1 or self.ncwo > self.nvir/self.ndoc):
                        self.ncwo = int(self.nvir/self.ndoc)
            else:
                self.ncwo = 0

        self.closed = (self.nbeta == (self.n_elec+self.mul-1)/2 and self.nalpha == (self.n_elec-self.mul+1)/2)
        
        self.ista = 0 #Use Static version of PNOF7: 0 = PNOF7 (Default), 1 = PNOF7s
        
        self.imod = 0 #Select versions of GNOFx: 0 = GNOF (Default), 1 = GNOFm, 2 = GNOFs
        
        self.nac = self.ndoc * (1 + self.ncwo)
        self.nbf5 = self.no1 + self.nac + self.nsoc   #JFHLY warning: nbf must be >nbf5 
        self.no0 = self.nbf - self.nbf5
        self.noptorb = self.nbf
        self.nvar = int(self.nbf*(self.nbf-1)/2) - int(self.no0*(self.no0-1)/2)
        self.nv = self.ncwo*self.ndoc
        self.HighSpin = False
        self.MSpin = 0
        self.alpha = 0.02
        
        if self.pair_doubles:
                # Kill singles completely (seniority-zero ansatz)
                self.singles = []

                # Keep only pair doubles
                self.doubles = self._build_pnof5e_omega()   
        else:
            self.singles, self.doubles = qml.qchem.excitations(self.n_elec, 2 * self.norb)
        
        print("Size Singles:",len(self.singles))
        print("Singles:",self.singles)
        print("Size Doubles:",len(self.doubles))
        print("Doubles:",self.doubles)
        
    
    # ---------------- Selecting the functional ----------------
    def _pnof(self, params, n_elec, norb, rdm1):
        if self.ipnof == 4:
            n, vecs, cj12, ck12, rdm1 = self._pnof4(params, n_elec, norb, rdm1)
        elif self.ipnof == 5:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof5(params, n_elec, norb, rdm1)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof5")
        elif self.ipnof == 7:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof7(params, n_elec, norb, rdm1)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof8") 
        elif self.ipnof == 8:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof8(params, n_elec, norb, rdm1)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof8")  
        else:
            raise ValueError(f"Unsupported PNOF level: {self.ipnof}")
        return n, vecs, cj12, ck12, rdm1
    
    # ---------------- Computing the energy ----------------
    def ene_nof(self, params, rdm1=None):
        
        E_nuc, J_MO, K_MO, H_core, n_elec, norb = self.E_nuc, self.J_MO, self.K_MO, self.H_core, self.n_elec, self.norb
        
        n, vecs, cj12, ck12, rdm1 = self._pnof(params, n_elec, norb, rdm1)
        
        E = self._calce(n,cj12,ck12,J_MO,K_MO,H_core)

        return E_nuc + E, rdm1, n, vecs, cj12, ck12
            
    # ---------------- Generate initial Parameters ----------------
    def _initial_params(self, params):
        if params is None:
            print("Params is none:", params)
            n_params = len(self.singles) + len(self.doubles)
            params = np.full(n_params, 0.1)
            print("Params after fill with 0.1 value:", params)
        else:
            print("Params is not none:", params)
        return params

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
    
    # ---------------- Filter pair ansatz ----------------
    def _build_pnof5e_omega(self):
        Omega_mo = self._build_pnof5e_omega_mo()

        Omega_spin = []
        for omega in Omega_mo:
            g = omega[0]          # strong orbital
            weak_orbs = omega[1:] # weak orbitals in Omega_g

            g_spin = [2*g, 2*g + 1]

            for q in weak_orbs:
                q_spin = [2*q, 2*q + 1]
                Omega_spin.append(g_spin + q_spin)
        return Omega_spin
    
    def _build_pnof5e_omega_mo(self):
        """
        Build Omega_g exactly as in PyNOF5 using ll / ul logic.
        Returns Omega in MO indices.
        """
        Omega = []
        
        for l in range(self.ndoc):
            g = self.no1 + l

            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo

            if ul > ll:
                weak = list(range(ll, ul))
                Omega.append([g] + weak)
            elif ul == ll:
                Omega.append([ul-1]+[ul])
                break
        return Omega
    
    def _JKH_MO_Full(self, C,H,I):
        #denmatj
        D = jnp.einsum('mi,ni->imn', C[:,0:self.nbf5], C[:,0:self.nbf5],optimize=True)
        #QJMATm
        J = jnp.tensordot(D, I, axes=([1,2],[2,3]))
        J_MO = jnp.tensordot(J, D,axes=((1,2),(1,2)))
        #QKMATm
        K = jnp.tensordot(D, I, axes=([1,2],[1,3]))
        K_MO = jnp.tensordot(K, D, axes=([1,2],[1,2]))
        #QHMATm
        H_core = jnp.tensordot(D, H, axes=([1,2],[0,1]))
        #breakpoint()
        return J_MO,K_MO,H_core
    
    def _calcorbg(self, y,n,cj12,ck12,C,H,I):

        Cnew = self._rotate_orbital(y,C)

        elag,_ = self._computeLagrange2(n,cj12,ck12,Cnew,H,I)

        grad = 4*(elag - elag.T)

        grads = np.zeros((self.nvar))

        k = 0
        for i in range(self.nbf5):
            for j in range(i+1,self.nbf):
                grads[k] = grad[i,j]
                k += 1
        grad = grads

        return grad
    
    def _computeLagrange2(self,n,cj12,ck12,C,H,I):
        
        H_mat = jnp.einsum("mi,mn,nj->ij",C,H,C[:,:self.nbf5],optimize=True)
        I_MO = jnp.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:self.nbf5],I,C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)

        cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
        ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
        
        elag = np.zeros((self.nbf,self.nbf))
        if(self.MSpin==0):
            # 2ndH/dy_ab
            
            #elag[:,:self.nbf5] +=  jnp.einsum('b,ab->ab',n,H_mat[:,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] +=  jnp.einsum('b,ab->ab',n[:self.nbf5],H_mat[:,:self.nbf5],optimize=True)

            # dJ_pp/dy_ab
            elag[:,:self.nbeta] +=  jnp.einsum('b,abbb->ab',n[:self.nbeta],I_MO[:,:self.nbeta,:self.nbeta,:self.nbeta],optimize=True)
            elag[:,self.nalpha:self.nbf5] +=  jnp.einsum('b,abbb->ab',n[self.nalpha:self.nbf5],I_MO[:,self.nalpha:self.nbf5,self.nalpha:self.nbf5,self.nalpha:self.nbf5],optimize=True)
            
            # C^J_pq dJ_pq/dy_ab 
            #elag[:,:self.nbf5] +=  jnp.einsum('bq,abqq->ab',cj12,I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] +=  jnp.einsum('bq,abqq->ab',cj12[:self.nbf5,:self.nbf5],I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)

            # -C^K_pq dK_pq/dy_ab 
            #elag[:,:self.nbf5] += -jnp.einsum('bq,aqbq->ab',ck12,I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] += -jnp.einsum('bq,aqbq->ab',ck12[:self.nbf5,:self.nbf5],I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
        else:
            raise RuntimeError("MSpin != 0 is not implemented yet")
        return elag, H_mat
    
    def _rotate_orbital(self, y,C):
        
        ynew = np.zeros((self.nbf,self.nbf))

        n = 0
        for i in range(self.nbf5):
            for j in range(i+1,self.nbf):
                ynew[i,j] =  y[n]
                ynew[j,i] = -y[n]
                n += 1

        U = expm(ynew)
        Cnew = jnp.einsum("mr,rp->mp",C,U,optimize=True)

        return Cnew
    
    def _calcorbe(self, y,n,cj12,ck12,C,H,I):

        Cnew = self._rotate_orbital(y,C)
        J_MO,K_MO,H_core = self._JKH_MO_Full(Cnew,H,I)
        E = self._calce(n,cj12,ck12,J_MO,K_MO,H_core)

        return E
    
    def _orbopt_adam(self, n,cj12,ck12,C,H,I):
        y = np.zeros((self.nvar))
        E = self._calcorbe(y, n,cj12,ck12,C,H,I)
        
        alpha = self.alpha
        beta1 = 0.7
        beta2 = 0.999

        y = np.zeros((self.nvar))

        m = 0.0 * y
        v = 0.0 * y
        vhat_max = 0.0 * y

        improved = False
        success = False
        best_E, best_C = E, C
        nit = 0
        for i in range(self.maxloop):
            nit += 1
            
            grad = self._calcorbg(y*0, n,cj12,ck12,C,H,I)
            if np.linalg.norm(grad) < 10**-4 and improved:
                success = True
                break

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            mhat = m / (1.0 - beta1**(i+1))
            vhat = v / (1.0 - beta2**(i+1))
            vhat_max = np.maximum(vhat_max, vhat)
            y = - alpha * mhat / (jnp.sqrt(vhat_max + 10**-8)) #AMSgrad
            C = self._rotate_orbital(y,C)
            
            E = self._calcorbe(y*0, n,cj12,ck12,C,H,I)
            print(f"Step = {i}, "
                    f"Etot = {float(self.E_nuc+E):.8f} Ha, "
                    f"E < best_E : {E < best_E}")
            if E < best_E:
                best_C = C
                best_E = E
                improved = True

        if not improved:
            self.alpha = self.alpha/10
            self.maxloop = self.maxloop + 30
            #print("      alpha ",self.alpha)
        
        return best_E,best_C,nit,success
        
    
    def _orbital_optimization(self, n, cj12,ck12, C,H,I):
        """
        Perform classical orbital optimization using PyNOF.
        H_MO and I_MO
        """
        
        E_orb, C_new, nit, success = self._orbopt_adam(
            n, cj12,ck12, C,H,I
        )
        
        if not success:
            print("Warning: orbital optimization did not fully converge")

        # Save updated orbitals
        self.C_MO = C_new

        return E_orb, C_new
    
    def _computeF_RC_CPU(self, J,K,n,H,cj12,ck12):

        # Matriz de Fock Generalizada
        F = jnp.zeros((self.nbf5,self.nbf,self.nbf))

        ini = 0
        if(self.no1>1):
            ini = self.no1

        # nH
        F += jnp.einsum('i,mn->imn',n,H,optimize=True)        # i = [1,nbf5]

        # # nJ
        
        # F[ini:self.nbeta,:,:] += jnp.einsum('i,imn->imn',n[ini:self.nbeta],J[ini:self.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
        # F[self.nalpha:self.nbf5,:,:] += jnp.einsum('i,imn->imn',n[self.nalpha:self.nbf5],J[self.nalpha:self.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

        # nJ
        F = F.at[ini:self.nbeta,:,:].add(
            jnp.einsum('i,imn->imn', n[ini:self.nbeta], J[ini:self.nbeta,:,:], optimize=True)
        )

        F = F.at[self.nalpha:self.nbf5,:,:].add(
            jnp.einsum('i,imn->imn', n[self.nalpha:self.nbf5], J[self.nalpha:self.nbf5,:,:], optimize=True)
        )
        
        # ---- Remove diagonal of cj12 block ----
        sub = cj12[ini:, ini:]
        sub = sub - jnp.diag(jnp.diag(sub))
        cj12 = cj12.at[ini:, ini:].set(sub)
        # C^J J
        F += jnp.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
        #F[ini:self.nbf5,:,:] -= jnp.einsum('ii,imn->imn',cj12[ini:self.nbf5,ini:self.nbf5],J[ini:self.nbf5,:,:],optimize=True) # quita i==j

        
        # ---- Remove diagonal of ck12 block ----
        sub = ck12[ini:, ini:]
        sub = sub - jnp.diag(jnp.diag(sub))
        ck12 = ck12.at[ini:, ini:].set(sub)
        # -C^K K
        F -= jnp.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
        #F[ini:self.nbf5,:,:] += jnp.einsum('ii,imn->imn',ck12[ini:self.nbf5,ini:self.nbf5],K[ini:self.nbf5,:,:],optimize=True) # quita i==j

        return F
    
    def _JKj_Full(self, C,I):
        #denmatj
        D = jnp.einsum('mi,ni->imn', C[:,0:self.nbf5], C[:,0:self.nbf5], optimize=True)
        #hstarj
        J = jnp.tensordot(D, I, axes=([1,2],[2,3]))
        #hstark
        K = jnp.tensordot(D, I, axes=([1,2],[1,3]))
        
        return J,K
    
    def _computeLagrange(self, F,C):

        G = jnp.einsum('imn,ni->mi',F,C[:,0:self.nbf5],optimize=True)

        #Compute Lagrange multipliers
        elag = jnp.zeros((self.nbf,self.nbf))
        elag = elag.at[0:self.noptorb,0:self.nbf5].add(
            jnp.einsum('mi,mj->ij',C[:,0:self.noptorb],G,optimize=True)[0:self.noptorb,0:self.nbf5]
        )
        
        # elag[0:self.noptorb,0:self.nbf5] = jnp.einsum('mi,mj->ij',C[:,0:self.noptorb],G,optimize=True)[0:self.noptorb,0:self.nbf5]

        return elag
    
    def _computeE_elec(self, H,C,n,elag):
        #EELECTRr
        E = 0

        E = E + jnp.einsum('ii',elag[:self.nbf5,:self.nbf5],optimize=True)
        E = E + jnp.einsum('i,mi,mn,ni',n[:self.nbeta],C[:,:self.nbeta],H,C[:,:self.nbeta],optimize=True)
        if(not self.HighSpin):
            E = E + jnp.einsum('i,mi,mn,ni',n[self.nbeta:self.nalpha],C[:,self.nbeta:self.nalpha],H,C[:,self.nbeta:self.nalpha],optimize=True)
        elif(self.HighSpin):
            E = E + 0.5*jnp.einsum('mi,mn,ni',C[:,self.nbeta:self.nalpha],H,C[:,self.nbeta:self.nalpha],optimize=True)

        E = E + jnp.einsum('i,mi,mn,ni',n[self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],H,C[:,self.nalpha:self.nbf5],optimize=True)

        return E
    
    def _computeLagrangeConvergency(self, elag):
        # Convergency

        sumdiff = np.sum(np.abs(elag-elag.T))
        maxdiff = np.max(np.abs(elag-elag.T))

        return sumdiff,maxdiff

    def _ENERGY1r(self, C,n,H,I,cj12,ck12):
        if(self.no1==0):
            elag,Hmat = self._computeLagrange2(n,cj12,ck12,C,H,I)
            E = jnp.einsum('ii',elag[:self.nbf5,:self.nbf5],optimize=True)
            E = E + jnp.einsum('i,ii',n[:self.nbf5],Hmat[:self.nbf5,:self.nbf5],optimize=True)
        else:
            J,K = self._JKj_Full(C,I)
            if(self.MSpin==0):
                F = self._computeF_RC_CPU(J,K,n,H,cj12,ck12)
            elif(not self.MSpin==0):
                raise RuntimeError("MSpin != 0 is not implemented yet")
            elag = self._computeLagrange(F,C)
            E = self._computeE_elec(H,C,n,elag)

        sumdiff,maxdiff = self._computeLagrangeConvergency(elag)
        
        return E,elag,sumdiff,maxdiff
    
    def run_scnofvqe(self, max_outer=30, tol=1e-6):
        """
        Self-consistent NOF-VQE loop:
        VQE amplitudes <-> orbital optimization
        """
        print("")

        print("-----------------------------------------------------")
        print("                Self-consistent NOF-VQE              ")
        print("-----------------------------------------------------")
        
        print("")
        
        print("functional:",self.functional)
        print("basis_set:",self.basis)
        E_old = None
        E_orb_old = None
        E_HF = None
        rdm1 = None
        
        # Computing atomic orbitals integrals
        S,H,I,E_nuc, wfn_mol, E_HF = self._ao_integrals(self.crd)
        self.E_nuc = E_nuc
        
        # Calling global parameters after computing integrals
        self._global_parameters(wfn_mol)
        
        # Initial parameters
        init_param = self.init_param
        
        # Initial orbitals
        C_MO = self.C_MO if self.C_MO is not None else None
        if C_MO is None:
            if self.gradient == "analytics":
                _, wfn_HF = psi4.energy("HF", return_wfn=True)
                C_HF = wfn_HF.Ca().np 
                
            else:
                _, C_HF, _, _, _ = qml.qchem.scf(wfn_mol)()
                
            C_MO = C_HF
        else:
        #     C_old = np.copy(C_MO)
        #     for i in range(self.ndoc):
        #         for j in range(self.ncwo):
        #             k = self.no1 + self.ndns + (self.ndoc - i - 1) * self.ncwo + j
        #             l = self.no1 + self.ndns + (self.ndoc - i - 1) + j*self.ndoc
        #             C_MO[:,k] = C_old[:,l]
            C_old = C_MO.copy()
            for i in range(self.ndoc):
                for j in range(self.ncwo):
                    k = self.no1 + self.ndns + (self.ndoc - i - 1) * self.ncwo + j
                    l = self.no1 + self.ndns + (self.ndoc - i - 1) + j*self.ndoc
                    C_MO = C_MO.at[:, k].set(C_old[:, l])
        C_MO = self._check_ortho(C_MO,S)
        
        print("HF energy:",E_HF)
            
        
        for it in range(max_outer):
            print(f"\n==== SC-NOFVQE Iteration {it} ====")
            
            # 1. Initial contions to perform optimization 
            #   (self._global_parameters() must be called before 
            #   to set the excitations into self._initial_params)
            self.init_param = self._initial_params(init_param)
            
            # 2. Build integrals with the current orbitals
            self.J_MO,self.K_MO,self.H_core = self._JKH_MO_Full(C_MO,H,I)
            
            # 3. Run VQE
            print("VQE Optimization")
            E_vqe, params, rdm1, n, vecs, cj12, ck12 = self.ene_vqe(self.init_param)
            print(f"Theta optimization VQE energy: {E_vqe:.10f} Ha")
            
            # 4. Orbital optimization
            print("Orbital Optimization (ADAM)")
            E_orb, C_new = self._orbital_optimization(n, cj12,ck12, C_MO,H,I)
            print(f"Orbital optimization energy: {self.E_nuc+E_orb:.10f} Ha")
            
            
            print("Updating initial parameters with optimal parameters",params)
            init_param = params
            
            
            # Convergence check (VQE and Orbital energy)
            if E_orb_old is not None:
                if abs(E_vqe - E_old) < tol and abs(E_orb - E_orb_old) < tol:
                    print("SC-NOFVQE converged (occupations + MO)")
                    break
            E_old = E_vqe
            E_orb_old = E_orb
            self.C_MO = C_new
            C_MO = C_new
            
        E, elag, _, _ = self._ENERGY1r(C_new,n,H,I,cj12,ck12)
        self.init_param = init_param
        print("")
        print("----------------")
        print(" Final Energies ")
        print("----------------")
        print("")
        print("       HF Total Energy = {:15.7f}".format(E_HF))
        print("Final NOF Total Energy = {:15.7f}".format(E_nuc + E))
        print("    Correlation Energy = {:15.7f}".format(E_nuc + E-E_HF))
        print("")
        print("")
        
        return E_nuc + E, params, rdm1, n, vecs, cj12, ck12, C_new, elag

    def _wrap_angles(self, p):
        p = np.asarray(p, dtype=float)
        # Map each parameter to (-pi, pi]
        return ((p + np.pi) % (2*np.pi)) - np.pi

    # ---------- measure 1-RDM on the circuit ----------
    #def _rdm1_from_circuit(self, params, n_elec, norb, region="eu-de", allow_fallback=False):
    def _rdm1_from_circuit(self, params, n_elec, norb, region="us-east", allow_fallback=False):
        max_retries = 10
        retry_delay = 5
        qubits = 2 * norb
        hf_state = [1] * n_elec + [0] * (qubits - n_elec)
        if self.dev in ["simulator", "hybrid"]:
            #print("Calling simulator or hybrid:")
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
            self._ansatz_2(theta, hf_state, qubits)
            return [qml.expval(op) for op in self._build_rdm1_ops(norb)]

        if self.pair_doubles:
            rdm1_ut = jnp.array(rdm1_qnode(params))  # length = norb*(norb+1)//2

            rdm1 = jnp.zeros((norb, norb))
            k = 0
            for p in range(norb):
                for q in range(p, norb):
                    rdm1 = rdm1.at[p, q].set(rdm1_ut[k])
                    rdm1 = rdm1.at[q, p].set(rdm1_ut[k])
                    k += 1
            assert rdm1.ndim == 2 and rdm1.shape[0] == rdm1.shape[1], \
            f"RDM1 has invalid shape {rdm1.shape}"
        else:
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
    
    def _pnof4(self, params, n_elec, norb, rdm1=None):
        # Fermi level
        F = int(n_elec / 2)

        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)
        n, vecs = self._get_no_on(rdm1, norb, self.pair_doubles)

        if self.pair_doubles:
            assert n.ndim == 1, f"Occupation numbers have wrong shape {n.shape}"
            assert vecs.ndim == 2, f"Orbital matrix has wrong shape {vecs.shape}"

        h = 1 - n
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
                    val = -jnp.sqrt(jnp.abs((n[q] * h[p] / S_F) * (n[p] - n[q] + n[q] * h[p] / S_F)))
                if p >= F and q < F:
                    val = -jnp.sqrt(jnp.abs((h[q] * n[p] / S_F) * (n[q] - n[p] + h[q] * n[p] / S_F)))
                if p >= F and q >= F:
                    val = jnp.sqrt(jnp.abs(n[q] * n[p]))
                Pi = Pi.at[q, p].set(val)
                
        n_outer = jnp.outer(n, n)
        cj12 = 2.0 * (n_outer - Delta)
        ck12 = n_outer - Delta - Pi
        return n, vecs, cj12, ck12, rdm1
    
    def _pnof5(self, params, n_elec, norb, rdm1=None):
        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)

        # Natural occupations and orbitals
        n, vecs = self._get_no_on(rdm1, norb, self.pair_doubles)
        
        if self.pair_doubles:
            assert n.ndim == 1, f"Occupation numbers have wrong shape {n.shape}"
            assert vecs.ndim == 2, f"Orbital matrix has wrong shape {vecs.shape}"

        
        # --- Build CJ and CK (PNOF5) ---
        cj12 = 2.0 * jnp.outer(n, n)
        ck12 = jnp.outer(n, n)
        
        if(self.MSpin==0 and self.nsoc>1):
            ck12 = ck12.at[self.nbeta:self.nalpha,self.nbeta:self.nalpha].set(2*jnp.outer(n[self.nbeta:self.nalpha],n[self.nbeta:self.nalpha]))

        # Pair structure (Omega_g)
        for l in range(self.ndoc):
            ldx = self.no1 + l         
            # inicio y fin de los orbitales acoplados a los fuertemente ocupados
            ll = self.no1 + self.ndns + (self.ndoc - l - 1)*self.ncwo
            ul = ll + self.ncwo

            n_strong = n[ldx]
            n_weak = n[ll:ul]

            # Remove Coulomb intra-pair
            cj12 = cj12.at[ldx,ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul,ldx].set(0.0)
            cj12 = cj12.at[ll:ul,ll:ul].set(0.0)

            # Exchange terms
            ck12 = ck12.at[ldx,ll:ul].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul,ldx].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul,ll:ul].set(
                -jnp.sqrt(jnp.outer(n_weak, n_weak))
            )
        return n, vecs, cj12, ck12, rdm1
    
    def _pnof7(self, params, n_elec, norb, rdm1=None):
        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)

        # Natural occupations and orbitals
        n, vecs = self._get_no_on(rdm1, norb, self.pair_doubles)
        
        if self.pair_doubles:
            assert n.ndim == 1, f"Occupation numbers have wrong shape {n.shape}"
            assert vecs.ndim == 2, f"Orbital matrix has wrong shape {vecs.shape}"

        if(self.ista==0):
            fi = n*(1-n)
            fi = fi.at[fi <= 0].set(0)
            fi = jnp.sqrt(fi)      
        else:
            fi = 2*n*(1-n)
        
        # --- Build CJ and CK (PNOF7) ---
        cj12 = 2.0 * jnp.outer(n, n)
        ck12 = jnp.outer(n, n) + jnp.outer(fi,fi)
        
        # Intrapair Electron Correlation
        
        if(self.MSpin==0 and self.nsoc>1):
            ck12 = ck12.at[self.nbeta:self.nalpha,self.nbeta:self.nalpha].set(2*jnp.outer(n[self.nbeta:self.nalpha],n[self.nbeta:self.nalpha]))

        # Pair structure (Omega_g)
        for l in range(self.ndoc):
            ldx = self.no1 + l         
            # inicio y fin de los orbitales acoplados a los fuertemente ocupados
            ll = self.no1 + self.ndns + (self.ndoc - l - 1)*self.ncwo
            ul = ll + self.ncwo

            n_strong = n[ldx]
            n_weak = n[ll:ul]

            # Remove Coulomb intra-pair
            cj12 = cj12.at[ldx,ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul,ldx].set(0.0)
            cj12 = cj12.at[ll:ul,ll:ul].set(0.0)

            # Exchange terms
            ck12 = ck12.at[ldx,ll:ul].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul,ldx].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul,ll:ul].set(
                -jnp.sqrt(jnp.outer(n_weak, n_weak))
            )
        return n, vecs, cj12, ck12, rdm1
    
    def _pnof8(self, params, n_elec, norb, rdm1=None):
        if rdm1 is None:
            rdm1 = self._rdm1_from_circuit(params, n_elec, norb)

        # Natural occupations and orbitals
        n, vecs = self._get_no_on(rdm1, norb, self.pair_doubles)

        # ----- PNOF8 auxiliary occupations -----
        h_cut = 0.02 * jnp.sqrt(2.0)
        n_d = jnp.zeros_like(n)

        for i in range(self.ndoc):

            idx = self.no1 + i
            ll = self.no1 + self.ndns + (self.ndoc - i - 1) * self.ncwo
            ul = ll + self.ncwo

            n_strong = n[idx]
            n_weak = n[ll:ul]

            h = 1.0 - n_strong
            coc = h / h_cut
            F = jnp.exp(-(coc**2))

            n_d = n_d.at[idx].set(n_strong * F)
            n_d = n_d.at[ll:ul].set(n_weak * F)

        n_d12 = jnp.sqrt(n_d)
        fi = n * (1 - n)
        fi = fi.at[fi <= 0].set(0)
        fi = jnp.sqrt(fi)
        
        # ----- Interpair Electron Correlation -----

        cj12 = 2.0 * jnp.outer(n, n)
        
        if self.ista == 0:
            ck12 = jnp.outer(n, n)

            ck12 = ck12.at[self.no1:self.nbeta, self.nalpha:].add(
                jnp.outer(fi[self.no1:self.nbeta], fi[self.nalpha:])
            )

            ck12 = ck12.at[self.nalpha:, self.no1:self.nbeta].add(
                jnp.outer(fi[self.nalpha:], fi[self.no1:self.nbeta])
            )

            ck12 = ck12.at[self.nalpha:, self.nalpha:].add(
                jnp.outer(fi[self.nalpha:], fi[self.nalpha:])
            )

            # ----- Intrapair Electron Correlation -----

            if self.MSpin == 0 and self.nsoc > 0:

                half = jnp.full((self.nsoc,), 0.5)

                ck12 = ck12.at[self.no1:self.nbeta, self.nbeta:self.nalpha].add(
                    0.5 * jnp.outer(fi[self.no1:self.nbeta], half)
                )

                ck12 = ck12.at[self.nbeta:self.nalpha, self.no1:self.nbeta].add(
                    0.5 * jnp.outer(half, fi[self.no1:self.nbeta])
                )

                ck12 = ck12.at[self.nbeta:self.nalpha, self.nalpha:].add(
                    jnp.outer(half, fi[self.nalpha:])
                )

                ck12 = ck12.at[self.nalpha:, self.nbeta:self.nalpha].add(
                    jnp.outer(fi[self.nalpha:], half)
                )

            if self.MSpin == 0 and self.nsoc > 1:
                ck12 = ck12.at[self.nbeta:self.nalpha, self.nbeta:self.nalpha].set(0.5)
        else:
            ck12 = jnp.outer(n,n) + jnp.outer(fi,fi)
        # ----- dynamic correlation terms -----

        ck12 = ck12.at[self.no1:self.nbeta, self.nalpha:].add(
            jnp.outer(n_d12[self.no1:self.nbeta], n_d12[self.nalpha:])
            - jnp.outer(n_d[self.no1:self.nbeta], n_d[self.nalpha:])
        )

        ck12 = ck12.at[self.nalpha:, self.no1:self.nbeta].add(
            jnp.outer(n_d12[self.nalpha:], n_d12[self.no1:self.nbeta])
            - jnp.outer(n_d[self.nalpha:], n_d[self.no1:self.nbeta])
        )

        ck12 = ck12.at[self.nalpha:, self.nalpha:].add(
            -jnp.outer(n_d12[self.nalpha:], n_d12[self.nalpha:])
            - jnp.outer(n_d[self.nalpha:], n_d[self.nalpha:])
        )

        # ----- Pair structure Omega_g -----

        for l in range(self.ndoc):

            ldx = self.no1 + l
            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo

            n_strong = n[ldx]
            n_weak = n[ll:ul]

            cj12 = cj12.at[ldx, ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul, ldx].set(0.0)
            cj12 = cj12.at[ll:ul, ll:ul].set(0.0)

            exch = jnp.sqrt(n_strong * n_weak)

            ck12 = ck12.at[ldx, ll:ul].set(exch)
            ck12 = ck12.at[ll:ul, ldx].set(exch)

            ck12 = ck12.at[ll:ul, ll:ul].set(
                -jnp.sqrt(jnp.outer(n_weak, n_weak))
            )

        return n, vecs, cj12, ck12, rdm1
    
    def _calce(self, n,cj12,ck12,J_MO,K_MO,H_core):
        E = 0
        n_test = n
        #breakpoint()
        if(self.MSpin==0):
            # 2H + J
            #E = E + 2*jnp.einsum('i,i',n,H_core,optimize=True)
            E = E + 2*jnp.einsum('i,i',n[:self.nbf5],H_core,optimize=True)
            
            E = E + jnp.einsum('i,i',n[:self.nbeta],jnp.diagonal(J_MO)[:self.nbeta],optimize=True)
            E = E + jnp.einsum('i,i',n[self.nalpha:self.nbf5],jnp.diagonal(J_MO)[self.nalpha:self.nbf5],optimize=True)
            #C^J JMO
            cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
            E = E + jnp.einsum('ij,ji->',cj12[:self.nbf5,:self.nbf5],J_MO,optimize=True) # sum_ij
            #C^K KMO
            ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
            E = E - jnp.einsum('ij,ji->',ck12[:self.nbf5,:self.nbf5],K_MO,optimize=True) # sum_ij
        else:
            print(f"Mspin:{self.MSpin} must be cero")
        return E

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

    def _vqe_opt_scipy(self, E_fn, method, params, max_iterations):
        """
        VQE optimization using:
            SLSQP, L-BFGS-B and COBYLA
        """

        # Define wrappers for scipy
        def E_scipy(x):
            x = jnp.array(x)  # ensure JAX array
            E_val, _, _, _, _, _ = E_fn(x)
            return float(E_val* self.energy_scale)
        
        # If running on a real/remote backend, do NOT supply an analytic jacobian to SciPy.
        # SciPy will use finite differences (function evaluations only) which is robust.
        use_analytic_jac = (method.lower() in ["slsqp", "l-bfgs-b"]) and (getattr(self, "dev", "simulator") == "simulator")
        
        #COBYLA optimizer doesn't support jacobian
        if use_analytic_jac:
            def grad_scipy(x):
                x = jnp.array(x)  # ensure JAX array
                grad_fn = lambda p: E_fn(p)[0]
                # This will call jax.grad - only safe on local simulator
                g = jax.grad(grad_fn)(x)
                return np.array(g, dtype=float)
            jac = grad_scipy
        else:
            # For remote devices or other methods, let SciPy approximate the jac via FD
            jac = None  # COBYLA doesn't use gradients
        
        bounds = [(-np.pi, np.pi) for _ in range(len(np.atleast_1d(params)))]

        iter_counter = {"i": 0}
        # this function prints the values per iterations, however, 
        # it computes the 1rdm two times more, so it is not recommended 
        # to use it in a real QC
        def callback(xk):
            x = jnp.array(xk)
            E_val, *_ = E_fn(x)

            if jac is not None:
                g = jac(xk)
                g_maxabs = np.max(jnp.abs(g))
                print(
                    f"Step = {iter_counter['i']}, "
                    f"Energy = {float(E_val):.8f} Ha, "
                    f"Gradient = {g_maxabs:.1e}"
                )
            else:
                print(
                    f"Step = {iter_counter['i']}, "
                    f"Energy = {float(E_val):.8f} Ha"
                )

            iter_counter["i"] += 1
            
        res = minimize(
            E_scipy,
            np.array(params, dtype=float),
            method=method.upper(),
            jac=jac,
            #jac=None,
            bounds=bounds,
            tol=self.conv_tol,
            options={
                "maxiter": max_iterations,
                "ftol": self.conv_tol,   # keep it
                "eps": 1e-8,             # critical
            },
            callback=callback,
        )
        
        res_x = np.asarray(res.x, dtype=float)
        res_x_wrapped = self._wrap_angles(res_x)

        E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(jnp.array(res_x_wrapped))
        return [E_val], [res.x], [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]
    
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
            method = self.opt_circ
        if max_iterations is None:
            max_iterations = self.max_iterations
        if params is None:
            params = self.init_param
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

    def ene_vqe(self, params=None):
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
        elif self.functional in ["pnof4","pnof5","pnof7","pnof8"]:
            E_history, params_history, rdm1_history, n_history, vecs_history, cj12_history, ck12_history = self._vqe(
                self.ene_nof, params
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
                #     self.ene_nof, self.opt_param, self.crd, max_iterations=1)
                E_hybrid, rdm1_hybrid, n_hybrid, vecs_hybrid, cj12_hybrid, ck12_hybrid = self.ene_nof(
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

                #######The energy is evaluated into the QC#######
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
        # last C_MO from run_scnofvqe
        C_MO = self.C_MO
        # loop over all atoms and Cartesian components
        for a in range(crds.shape[0]):
            for xyz in range(3):
                # displaced geometries (full copies)
                crds_plus = crds.copy()
                crds_minus = crds.copy()
                # displaced geometries
                crds_plus[a, xyz] = crds[a, xyz] + d_shift
                crds_minus[a, xyz] = crds[a, xyz] - d_shift

                if self.functional not in ["pnof4","pnof5","pnof7","pnof8"]:
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
                    #Computing the integral using crds_plus
                    mol_plus = self._molecule_pnl(crds_plus)
                    _,H_plus,I_plus,E_nuc_plus, _,_ = self._ao_integrals_pnl(crds_plus, mol_plus)
                    J_MO_plus,K_MO_plus,H_core_plus = self._JKH_MO_Full(C_MO,H_plus,I_plus)
                    n, _, cj12, ck12, _ = self._pnof(params, self.n_elec, self.norb, rdm1=rdm1_opt)
                    E_elc_plus = self._calce(n,cj12,ck12,J_MO_plus,K_MO_plus,H_core_plus)
                    E_plus = E_nuc_plus + E_elc_plus
                    
                    #Computing the integral using crds_minus
                    mol_minus = self._molecule_pnl(crds_minus)
                    _,H_minus,I_minus,E_nuc_minus, _,_ = self._ao_integrals_pnl(crds_minus, mol_minus)
                    J_MO_minus,K_MO_minus,H_core_minus = self._JKH_MO_Full(C_MO,H_minus,I_minus)
                    n, _, cj12, ck12, _ = self._pnof(params, self.n_elec, self.norb, rdm1=rdm1_opt)
                    E_elc_minus = self._calce(n,cj12,ck12,J_MO_minus,K_MO_minus,H_core_minus)
                    E_minus = E_nuc_minus + E_elc_minus
                    
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

                if self.functional not in ["pnof4","pnof5","pnof7","pnof8"]:
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
                    E_plus, p_plus, _, _, _, _, _ = self._vqe(self.ene_nof, p_start_plus)
                    self.E_nuc, self.h_MO, self.I_MO, self.n_elec, self.norb = self._mo_integrals(crds_minus)
                    E_minus, p_minus, _, _, _, _, _ = self._vqe(self.ene_nof, p_start_minus)

                # Optional: update warm-start for the next coordinate
                if warm_start:
                    params_p = p_plus[-1]  # carry forward the latest parameters
                    params_m = p_minus[-1]  # carry forward the latest parameters

                grad[a, xyz] = (E_plus[-1] - E_minus[-1]) / (2 * d_shift)

        return grad
    
    def _nuclear_gradient_analytics(self, n,C,cj12,ck12,elag):
        
        mints = psi4.core.MintsHelper(self.wfn.basisset())
        
        RDM1 = 2*jnp.einsum('p,mp,np->mn',n[:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        lag = 2*jnp.einsum('mq,qp,np->mn',C,elag,C,optimize=True)
        
        grad = jnp.zeros((self.natoms,3))
        
        grad += jnp.array(self.mol.nuclear_repulsion_energy_deriv1())
        
        for i in range(self.natoms):
            dSx,dSy,dSz = jnp.array(mints.ao_oei_deriv1("OVERLAP",i))
            #breakpoint()
            grad = grad.at[i,0].add(-jnp.einsum('mn,mn->',lag,dSx,optimize=True))
            grad = grad.at[i,1].add(-jnp.einsum('mn,mn->',lag,dSy,optimize=True))
            grad = grad.at[i,2].add(-jnp.einsum('mn,mn->',lag,dSz,optimize=True))
            #breakpoint()
            dTx,dTy,dTz = jnp.array(mints.ao_oei_deriv1("KINETIC",i))
            grad = grad.at[i,0].add(jnp.einsum('mn,mn->',RDM1,dTx,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum('mn,mn->',RDM1,dTy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum('mn,mn->',RDM1,dTz,optimize=True))

            dVx,dVy,dVz = jnp.array(mints.ao_oei_deriv1("POTENTIAL",i))
            grad = grad.at[i,0].add(jnp.einsum('mn,mn->',RDM1,dVx,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum('mn,mn->',RDM1,dVy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum('mn,mn->',RDM1,dVz,optimize=True))
        
        cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
        ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
        
        #RDM2 = jnp.einsum('pq,mp,np,sq,lq->mnsl',cj12,C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 = jnp.einsum('pq,mp,np,sq,lq->mnsl',cj12[:self.nbf5,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 += jnp.einsum('p,mp,np,sp,lp->mnsl',n[:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],optimize=True)
        RDM2 += jnp.einsum('p,mp,np,sp,lp->mnsl',n[self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],optimize=True)
        #RDM2 -= jnp.einsum('pq,mp,lp,sq,nq->mnsl',ck12,C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 -= jnp.einsum('pq,mp,lp,sq,nq->mnsl',ck12[:self.nbf5,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)

        for i in range(self.natoms):
            derix,deriy,deriz = jnp.array(mints.ao_tei_deriv1(i))
            grad = grad.at[i,0].add(jnp.einsum("mnsl,mnsl->",RDM2,derix,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum("mnsl,mnsl->",RDM2,deriy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum("mnsl,mnsl->",RDM2,deriz,optimize=True))
        return grad
    
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
    #xyz_file = "lih_bohr.xyz"
    #xyz_file = "lih_bohr.xyz"
    #functional=sys.argv[2]
    functional="pnof8"
    #functional="vqe"
    conv_tol=1e-6
    #init_param=0.1
    init_param=None
    basis=sys.argv[2]
    #basis='sto-3g'
    #basis='6-31G'
    max_iterations=200
    gradient=sys.argv[3]
    #gradient="analytics"
    #gradient="df_fedorov"
    d_shift=1e-4
    C_MO = None
    dev="simulator"
    #opt_circ="sgd"
    opt_circ="slsqp"
    n_shots=10000
    optimization_level=3
    resilience_level=0
    # arg = sys.argv[4].lower()
    # if arg == "true":
    #     pair_double = True
    # elif arg == "false":
    #     pair_double = False
    # else:
    #     raise ValueError("pair_double must be True or False")
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
    # SCF-NOFVQE G.S.Energy
    E_min, params_opt, rdm1_opt, n, vecs, cj12, ck12, C_opt, elag = cal.run_scnofvqe()
    print("Min Ene VQE and param:", E_min, params_opt)
    print("ON",2*n)
    # Nuclear Gradient (Analytic)
    grad = cal._nuclear_gradient_analytics(n,C_opt,cj12,ck12,elag)
    print(f"Nuclear gradient ({gradient}):\n", grad)
    print(f"Nuclear gradient norm:\n", np.linalg.norm(grad))