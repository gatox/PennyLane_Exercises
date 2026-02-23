import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

from scipy.linalg import eigh, fractional_matrix_power

import pynof

symbols = ["Li", "H"]
geometry = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 3.015]])  # bohr

mol = qml.qchem.Molecule(
    symbols,
    geometry,
    basis_name="sto-3g",
    charge=0,
    mult=1,
    unit="bohr"
)

# PennyLane MO coefficients
_, coeffs, _, h_core, rep_tensor = qml.qchem.scf(mol)()

C_PL = coeffs
print("PennyLane MO coefficients", C_PL)

one = qml.math.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
two = qml.math.swapaxes(
    qml.math.einsum(
        "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, rep_tensor, coeffs, coeffs
    ),
    1,
    3,
)
core_constant = qml.qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)()

S_matrix = qml.qchem.overlap_matrix(mol.basis_set)()
X_matrix = fractional_matrix_power(S_matrix, -0.5)
breakpoint()


E_nuc, h_MO, I_MO = qml.qchem.electron_integrals(mol)()

def make_pairs(n_orb, n_elec):
    n_pairs = n_elec // 2
    return [list(range(2*i, 2*i+2)) for i in range(n_pairs)]

def _filter_pair_doubles(doubles):
    pair_doubles = []
    for d in doubles:
        i, j, a, b = d
        if (j == i + 1) and (b == a + 1):
            if (i % 2 == 0) and (a % 2 == 0):
                pair_doubles.append(d)
    return pair_doubles










# print("I in ao PL",I_MO_PL)

# # PennyLane Hermiticity
# assert np.allclose(I_MO_PL, I_MO_PL.transpose(2,3,0,1), atol=1e-10)

# # Exchange symmetry
# assert np.allclose(I_MO_PL, I_MO_PL.transpose(1,0,3,2), atol=1e-10)

print("PennyLane I_MO_PL (pq|pq):")
for i in range(I_MO_PL.shape[0]):
    print(f"p={i}: {(I_MO_PL[i,i,i,i]): .8f}")

mole = pynof.molecule("""
Units Bohr
0 1
  Li  0.0000   0.000   0.000
  H  0.0000   0.000  3.015
""")
basis= 'sto-3g'
p = pynof.param(mole,basis)
p.ipnof = 4
p.RI = False

S_ao, _, _, H_ao, I_ao, b_mnl, _ = pynof.compute_integrals(p.wfn,mole,p)

#print("I in ao pynof",I_ao)

assert np.allclose(I_MO_from_AO, I_MO_PL, atol=1e-10)


# _, C = eigh(H_ao, S_ao)
# C_old = pynof.check_ortho(C,S_ao,p)

# print("Pynof MO coefficients", C_old)

# print("I_ao.shape:",I_ao.shape)
# print("C_old.shape:",C_old.shape)
# print("b_mnl",b_mnl)

# I_ao_chem = np.transpose(I_ao, (0, 2, 1, 3))
# # (m n l s) → (m l n s)

# I_MO_from_PyNOF = np.einsum(
#     "mp,nq,lr,st,mnls->pqrs",
#     C_PL, C_PL, C_PL, C_PL,
#     I_ao_chem,
#     optimize=True
# )

# # Hermiticity
# assert np.allclose(I_MO_from_PyNOF,
#                    I_MO_from_PyNOF.transpose(2,3,0,1),
#                    atol=1e-10)

# # Exchange
# assert np.allclose(I_MO_from_PyNOF,
#                    I_MO_from_PyNOF.transpose(1,0,3,2),
#                    atol=1e-10)

# # Compare to PennyLane
# assert np.allclose(I_MO_from_PyNOF, I_MO_PL, atol=1e-8)



# # I_MO = np.einsum(
# #     "mp,ns,lq,sr,mnls->psqr",
# #     C_old, C_old, C_old, C_old, I_ao,
# #     optimize=True
# # )

# # # PennyLane Hermiticity
# # assert np.allclose(I_MO, I_MO.transpose(2,3,0,1), atol=1e-10)

# # # Exchange symmetry
# # assert np.allclose(I_MO, I_MO.transpose(1,0,3,2), atol=1e-10)



# # # Diagonal Coulomb sanity check
# # print("Psi4 → PennyLane (pq|pq):")
# # for p in range(I_MO.shape[0]):
# #     print(f"p={p}: {I_MO[p,p,p,p]: .8f}")

# I_ao = np.transpose(I_ao, (0, 2, 1, 3))

# import itertools

# print("################# Testing brute-force ##################")

# ref = np.array([I_MO_PL[p,p,p,p] for p in range(I_MO_PL.shape[0])])

# def test_mapping(label, I_try):
#     vals = np.array([I_try[p,p,p,p] for p in range(I_try.shape[0])])
#     if np.allclose(vals, ref, atol=1e-6):
#         print("MATCH FOUND:", label)
#         print(vals)
#         return True
#     return False


# found = False

# for C_use, C_label in [(C_old, "C"), (C_old.T, "C.T")]:
#     for perm in itertools.permutations([0,1,2,3]):
#         # permute AO indices
#         I_perm = np.transpose(I_ao, perm)

#         # standard chemist AO→MO
#         I_try = np.einsum(
#             "mp,nq,lr,st,mnls->pqrs",
#             C_use, C_use, C_use, C_use, I_perm,
#             optimize=True
#         )

#         # test all PennyLane-style reorderings
#         for out_perm in itertools.permutations([0,1,2,3]):
#             I_out = np.transpose(I_try, out_perm)
#             label = f"C={C_label}, AOperm={perm}, MOperm={out_perm}"
#             if test_mapping(label, I_out):
#                 found = True
#                 break
#         if found:
#             break
#     if found:
#         break

# if not found:
#     print("❌ No mapping found")

