#!/usr/bin/env python3
import sys
import numpy as np
from pyscf import gto

# -----------------------------
# Read ORCA EXT input
# -----------------------------
def read_extinp(filename):
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]

    xyzfile = lines[0].split()[0]                # first token only
    charge = int(lines[1].split()[0])           # remove comment
    multiplicity = int(lines[2].split()[0])     # remove comment
    ncores = int(lines[3].split()[0])
    do_grad = int(lines[4].split()[0]) if len(lines) > 4 else 1

    return xyzfile, charge, multiplicity, do_grad

# -----------------------------
# Read XYZ coordinates
# -----------------------------
def read_xyz(xyzfile):
    with open(xyzfile) as f:
        lines = [l.strip() for l in f if l.strip()]
    nat = int(lines[0])
    atoms = [l.split() for l in lines[2:2+nat]]
    return [(a[0], [float(x) for x in a[1:]]) for a in atoms]

def read_natoms(xyzfile):
    with open(xyzfile) as f:
        lines = [l.strip() for l in f if l.strip()]
    nat = int(lines[0])
    atoms = [l.split() for l in lines[2:2+nat]]
    return nat


# -----------------------------
# Write ORCA .engrad file
# -----------------------------
def write_engrad(base, energy, grad):
    """
    Write ORCA ExtOpt .engrad file in strict format:
    - Includes all # comment lines exactly as expected
    - Gradients in Eh/Bohr, one number per line
    """
    natoms = grad.shape[0]
    fname = f"{base}_EXT.engrad"
    with open(fname, 'w') as f:
        f.write("#\n")
        f.write("# Number of atoms: must match the XYZ\n")
        f.write("#\n")
        f.write(f"{natoms}\n")
        f.write("#\n")
        f.write("# The current total energy in Eh\n")
        f.write("#\n")
        f.write(f"{energy:.12f}\n")
        f.write("#\n")
        f.write("# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, etc.\n")
        f.write("#\n")
        # Write gradients flattened (row-major), one per line
        for g in grad.reshape(-1):
            f.write(f"{g:.12e}\n")

def write_orca_hess(output_filename, hessian_matrix, natoms):
    """
    Writes the given Hessian matrix to a file in ORCA .hess format.
    
    Format:
    $hessian
    <natoms>
    <natoms * 3>
    <full 3N x 3N matrix, space-delimited>
    """
    n_coords = natoms * 3
    
    # Ensure matrix is the correct shape
    if hessian_matrix.shape != (n_coords, n_coords):
        print(f"Error: Hessian matrix shape is {hessian_matrix.shape}, but expected ({n_coords}, {n_coords})")
        return

    try:
        with open(output_filename, 'w') as f:
            # Write header
            f.write("$hessian\n")
            f.write(f"{natoms}\n")
            f.write(f"{n_coords}\n")
        
        # Append the matrix using numpy.savetxt for clean, aligned formatting
        # We open in 'a' (append) mode to add to the file we just created.
        with open(output_filename, 'a') as f:
            np.savetxt(f, hessian_matrix, fmt='%20.10f')
            
        print(f"Successfully wrote ORCA Hessian to {output_filename}")
        
    except Exception as e:
        print(f"Error writing Hessian file: {e}")

def get_masses(symbols):
         """Return list of atomic masses (in a.m.u.) for given atom symbols."""
         mass_table = {
        "H": 1.00784, "He": 4.002602,
        "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
        "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
        "K": 39.0983, "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845,
        "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971,
        "Br": 79.904, "Kr": 83.798,
        "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.90550,
        "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293,
        "Cs": 132.90545196, "Ba": 137.327,
        "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
        "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668,
        "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.592,
        "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0,
        "Fr": 223.0, "Ra": 226.0,
        "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0, "Bk": 247.0,
        "Cf": 251.0, "Es": 252.0, "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 262.0,
        "Rf": 267.0, "Db": 270.0, "Sg": 271.0, "Bh": 270.0, "Hs": 277.0, "Mt": 276.0, "Ds": 281.0, "Rg": 282.0, "Cn": 285.0,
        "Nh": 286.0, "Fl": 289.0, "Mc": 290.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0
         }
             
         return [mass_table[s] for s in symbols]
# -----------------------------
# Run PySCF energy + gradient
# -----------------------------
def run_pyscf(xyzfile, charge, mult, do_grad,hess=False):
    import numpy as np
    from gpu4pyscf.dft import rks
    from gpu4pyscf import  grad

    atoms = read_xyz(xyzfile)

    mol = gto.M(
        atom=atoms,
        basis={
            'C': 'aug-cc-pvdz',
            'N': 'aug-cc-pvdz',
            'H': 'aug-cc-pvdz',
            'Br': 'aug-cc-pvdz',
            'O': 'aug-cc-pvdz',
            'Mn': gto.basis.load('stuttgart_rsc', 'Mn')
        },
        ecp={'Mn': gto.load_ecp('stuttgart_rsc', 'Mn')},
        unit="angstrom",
        charge=charge,
        spin=mult-1,
        verbose=3
    )

    # Create GPU RKS object
    mf = rks.RKS(mol, xc='M06L').density_fit()

    # Optional solvent (may need CPU fallback if not supported on GPU)
    mf = mf.SMD()
    mf.with_solvent.method = 'SMD'
    mf.with_solvent.solvent = '1,4-dioxane'

    mf.max_cycle = 1000
    mf.grids.level = 6
    mf.conv_tol = 1e-7

    # Energy calculation
    energy = mf.kernel()

    # Gradient calculation using GPU4PySCF
    if do_grad:
        grad_calc = mf.nuc_grad_method().kernel()  # GPU-compatible
    else:
        grad_calc = np.zeros((mol.natm, 3))

    if hess:
        hess_matx=mf.Hessian().kernel()
    else:
        hess_matx = np.zeros((3*mol.natm, 3*mol.natm))
    return energy, grad_calc, hess_matx


# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    extinp = sys.argv[1]
    base = extinp.split('_EXT')[0]
    ext = sys.argv[2]
    value = ext.split('=')[1]
    H = value.lower() == "true"

    xyzfile, charge, mult, do_grad = read_extinp(extinp)
    energy, grad,hess_mat = run_pyscf(xyzfile, charge, mult, do_grad, hess=H)
    write_engrad(base, energy, grad)
    natoms=read_natoms(xyzfile)

    if H:
        hess_mat = hess_mat.transpose(0, 2, 1, 3).reshape(3*natoms, 3*natoms)
        # Read coordinates, elements, masses from xyzfile
        elems, coords = [], []
        with open(xyzfile) as f:
            lines = f.readlines()[2:]  # skip first 2 lines of XYZ
            for line in lines:
                parts = line.split()
                elems.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
        coords = np.array(coords)

        # You can get masses using ASH built-in function or manually:
        masses = get_masses(elems)
        print(masses)
#        masses = [get_masses(elem) for elem in elems]
        masses = np.array(masses)
        hessatoms = list(range(len(elems)))
        hess_file = base + ".hess"
        from ash import *
#        import ash.modules.module_freq
        hessian=hess_mat
        hesscoords=coords.flatten()
        print(hesscoords)
        hessmasses=masses.flatten()
        print(hessmasses)
        hesselems=elems
        dipole_derivs = np.zeros((3 * natoms, 3))

        if ash.modules.module_freq.detect_linear(coords=coords,elems=elems) == True:
           TRmodenum=5
        else:
           TRmodenum=6

        frequencies, nmodes, evectors, mode_order = ash.modules.module_freq.diagonalizeHessian(hesscoords,hessian,hessmasses,hesselems, TRmodenum=TRmodenum,projection=True)

        IR_intens_values = ash.modules.module_freq.calc_IR_Intensities(hessmasses,evectors,dipole_derivs)

# Raman activities if polarizabilities available
#        Raman_activities, depolarization_ratios = ash.modules.module_freq.calc_Raman_activities(hessmasses,evectors,polarizability_derivs)

# Print out Freq output.
        ash.modules.module_freq.printfreqs(frequencies,len(hessatoms),TRmodenum=TRmodenum, intensities=IR_intens_values)

        frag = Fragment(xyzfile=xyzfile)

# Print thermochemistry for new vibrational frequencies
        thermodict = ash.modules.module_freq.thermochemcalc(frequencies,hessatoms, frag, frag.mult)
        total_gibbs_energy= thermodict["Gcorr"] + energy
        print("##############################################################")
        print(f"Final Gibbs Energy:{total_gibbs_energy} Eh")
        print("##############################################################")
        
        
        ash.modules.module_freq.write_hessian(hessian,hessfile=hess_file)
        ash.modules.module_freq.printdummyORCAfile(elems=elems,coords=coords,vfreq=frequencies,evectors=evectors,nmodes=nmodes,hessfile=hess_file)

        print(f"✅ Wrote ORCA-style Hessian to {hess_file}")
