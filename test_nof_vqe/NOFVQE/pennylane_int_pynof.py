import pynof
import numpy as np
import pennylane as qml
import sys

# mol = pynof.molecule("""
# units bohr 
# 0 1
#   Li  0.0000  0.0000 0.0000 
#   H  0.0000  0.0000 3.02356 
# """)
xyz_file = sys.argv[1]
pnl = sys.argv[2]
if pnl == "True":
    print("Using pennylane Integrals")
    pnl = True
elif pnl == "False":
    print("Using psi4 Integrals")
    pnl = False
else:
    print("Not pnl well defined")
units, charge, multiplicity, symbols, geometry, mol = pynof.read_mol(xyz_file)

mol_pnl = qml.qchem.Molecule(
    symbols,
    geometry,
    basis_name="sto-3g",
    charge=0,
    mult=1,
    unit="bohr"
)

p = pynof.param(mol,"sto-3g")

p.ipnof=5

p.RI = False
#p.gpu = True
wfn = p.wfn

if not pnl:
    E,C,gamma,fmiug0 = pynof.compute_energy(mol,p, guess="HF")
elif pnl:
    E,C,gamma,fmiug0 = pynof.compute_energy(mol_pnl,p, guess="HF", pnl=True)


# # print("Nuclear norm:",np.linalg.norm(g))
