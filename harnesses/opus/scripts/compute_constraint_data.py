#!/usr/bin/env python
"""Densities used by the physical-constraint tests (CPU PySCF, ωB97M-V/def2-QZVPPD).

  O2_triplet.h5  : open-shell grid (spin-symmetry test)
  H2O.h5         : closed-shell grid (uniform-coordinate-scaling test of exchange)
  AE18_grids.h5  : the 18 AE18 atoms (H-Ar), density from a (250,974) SCF evaluated on both the
                   (99,590) and (250,974) grids (grid-sensitivity test, as in Duston et al. 2026)
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from pyscf import gto, dft

from dftsr.config import CONSTRAINT_DIR, BASIS
from dftsr.dataset import resolve_geometry, read_xyz

AE18 = ['11_H_AE18', '12_He_AE18', '13_Li_AE18', '14_Be_AE18', '15_B_AE18', '16_C_AE18', '17_N_AE18',
        '18_O_AE18', '19_F_AE18', '20_Ne_AE18', '21_Na_AE18', '22_Mg_AE18', '23_Al_AE18', '24_Si_AE18',
        '25_P_AE18', '26_S_AE18', '27_Cl_AE18', '28_Ar_AE18']


def scf(atom, charge, spin, atom_grid=None, level=3):
    mol = gto.M(atom=atom, basis=BASIS, charge=charge, spin=spin, verbose=0, max_memory=32000)
    mf = dft.UKS(mol)
    mf.xc = 'wb97m-v'
    mf.nlcgrids.prune = dft.gen_grid.sg1_prune
    if atom_grid:
        mf.grids.atom_grid = atom_grid
        mf.grids.prune = None
    else:
        mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf


def features(mol, mf, grids):
    out = []
    for s in range(2):
        ao = dft.numint.eval_ao(mol, grids.coords, deriv=1)
        out.append(dft.numint.eval_rho2(mol, ao, mf.mo_coeff[s], mf.mo_occ[s], xctype='MGGA'))
    return np.array(out), grids.weights


def save(f, grp, rho, w, coords=None, **attrs):
    g = f.create_group(grp)
    g['rho'] = rho
    g['weights'] = w
    if coords is not None:
        g['coords'] = coords
    for k, v in attrs.items():
        g.attrs[k] = v


def main():
    os.makedirs(CONSTRAINT_DIR, exist_ok=True)
    with h5py.File(os.path.join(CONSTRAINT_DIR, 'constraints.h5'), 'w') as f:
        mol, mf = scf('O 0 0 0; O 0 0 1.2075', 0, 2)
        rho, w = features(mol, mf, mf.grids)
        save(f, 'O2_triplet', rho, w, mf.grids.coords, E_tot=mf.e_tot)
        print('O2', mf.e_tot, rho.shape)

        atom, ch, mult = read_xyz(resolve_geometry('01_water-ammonia_1p0_monA_A21x12'))
        mol, mf = scf(atom, ch, mult - 1)
        rho, w = features(mol, mf, mf.grids)
        save(f, 'H2O', rho, w, mf.grids.coords, E_tot=mf.e_tot)
        print('H2O', mf.e_tot, rho.shape)

        for name in AE18:
            atom, ch, mult = read_xyz(resolve_geometry(name))
            mol, mf = scf(atom, ch, mult - 1, atom_grid=(250, 974))
            for tag, ag in (('fine', (250, 974)), ('coarse', (99, 590))):
                g = dft.gen_grid.Grids(mol)
                g.atom_grid = ag
                g.prune = None
                g.build()
                rho, w = features(mol, mf, g)
                save(f, f'AE18/{name}/{tag}', rho, w, E_tot=mf.e_tot)
            print(name, mf.e_tot, mf.converged)


if __name__ == '__main__':
    main()
