#!/usr/bin/env python
"""Compute ωB97M-V/def2-QZVPPD reference densities for MGCDB84 and save them for non-SCF evaluation.

Protocol per molecule (GPU4PySCF, one process per GPU):
  1. RI-JK (def2-universal-jkfit) SCF with ωB97M-V, PySCF grid level 3, SG-1 VV10 grid.
  2. One exact-integral (no RI) evaluation of every energy component at the converged density
     (error vs a full non-RI SCF is 2nd order in the density error: <~3e-6 Ha in benchmarks).
  3. rho, grad rho, tau for each spin on the XC grid -> HDF5 (float64), plus MOs for SCF warm starts.

Stored energy components (Hartree), all evaluated with exact integrals:
  E_core (T + V_ne), E_J, E_nuc, E_sl (libxc ωB97M-V semilocal part on the grid), E_nlc (VV10, b=6, C=0.01),
  E_xx (full-range HF exchange), E_xx_lr (long-range HF exchange, ω=0.3),
  E_tot = E_core + E_J + E_nuc + E_sl + E_nlc + 0.15 E_xx + 0.85 E_xx_lr.

Usage (resumable; run one per GPU; workers coordinate through lock files):
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_densities.py --splits train,validation,test
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from dftsr.config import DENSITY_DIR, BASIS, GRID_LEVEL, OMEGA
from dftsr.dataset import load_reactions, molecules_for, all_molecules, resolve_geometry, read_xyz

CX_SR = 0.15


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(name, lockdir):
    path = os.path.join(lockdir, name + '.lock')
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            host, pid = open(path).read().split()[:2]
            if host == socket.gethostname() and not pid_alive(int(pid)):
                os.remove(path)          # stale lock from a dead worker
                return claim(name, lockdir)
        except Exception:
            pass
        return False
    os.write(fd, f'{socket.gethostname()} {os.getpid()} {time.time()}'.encode())
    os.close(fd)
    return True


def is_done(name):
    p = os.path.join(DENSITY_DIR, name + '.h5')
    if not os.path.exists(p):
        return False
    try:
        with h5py.File(p, 'r') as f:
            return 'E_tot' in f.attrs
    except Exception:
        return False


def run_scf(mol, mult, log):
    from gpu4pyscf import dft
    from pyscf.dft import gen_grid
    K = dft.RKS if mult == 1 else dft.UKS

    def make(df=True):
        mf = K(mol, xc='wb97m-v')
        if df:
            mf = mf.density_fit(auxbasis='def2-universal-jkfit')
        mf.grids.level = GRID_LEVEL
        if max(mol.atom_charges()) <= 18:      # SG-1 is only defined for H-Ar (as in SyFES / MGCDB84)
            mf.nlcgrids.prune = gen_grid.sg1_prune
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf.verbose = 0
        return mf

    mf = make()
    mf.kernel()
    attempts = ['diis']
    if not mf.converged:
        dm0 = mf.make_rdm1()
        mf = make()
        mf.level_shift = 0.25
        mf.damp = 0.5
        mf.max_cycle = 300
        mf.kernel(dm0=dm0)
        attempts.append('levelshift+damp')
    if not mf.converged:
        try:
            dm0 = mf.make_rdm1()
            mf2 = make().newton()
            mf2.max_cycle = 100
            mf2.kernel(dm0=dm0)
            mf = mf2
            attempts.append('newton')
        except Exception as e:  # noqa
            log(f'newton failed: {e}')
    return mf, attempts, make(df=False)


def exact_components(mol, mf, mfx, mult):
    import cupy as cp
    dm = mf.make_rdm1()
    dmt = dm if mult == 1 else dm[0] + dm[1]
    mfx.grids = mf.grids
    mfx.nlcgrids = mf.nlcgrids
    h = mfx.get_hcore()
    e_core = float(cp.einsum('ij,ji', h, dmt))
    vj = mfx.get_j(mol, dmt)
    e_j = 0.5 * float(cp.einsum('ij,ji', vj, dmt))
    vk = mfx.get_k(mol, dm)
    vkl = mfx.get_k(mol, dm, omega=OMEGA)
    if mult == 1:
        e_xx = -0.25 * float(cp.einsum('ij,ji', vk, dm))
        e_xl = -0.25 * float(cp.einsum('ij,ji', vkl, dm))
    else:
        e_xx = -0.5 * float(cp.einsum('sij,sji', vk, dm))
        e_xl = -0.5 * float(cp.einsum('sij,sji', vkl, dm))
    del vk, vkl, vj
    ni = mfx._numint
    if mult == 1:
        _, e_sl, _ = ni.nr_rks(mol, mfx.grids, 'wb97m-v', dm)
    else:
        _, e_sl, _ = ni.nr_uks(mol, mfx.grids, 'wb97m-v', dm)
    e_sl = float(np.sum(cp.asnumpy(cp.asarray(e_sl))))
    e_nlc = float(ni.nr_nlc_vxc(mol, mfx.nlcgrids, 'wb97m-v', dmt)[1])
    e_nuc = float(mol.energy_nuc())
    e_tot = e_core + e_j + e_nuc + e_sl + e_nlc + CX_SR * e_xx + (1 - CX_SR) * e_xl
    return dict(E_core=e_core, E_J=e_j, E_nuc=e_nuc, E_sl=e_sl, E_nlc=e_nlc, E_xx=e_xx,
                E_xx_lr=e_xl, E_tot=e_tot)


def grid_features(mol, mf, mult, block=32768):
    """Return rho array (nspin, 5, N) [rho, dx, dy, dz, tau] per spin, weights, coords (numpy)."""
    import cupy as cp
    from gpu4pyscf.dft import numint as gni
    coords = mf.grids.coords
    weights = mf.grids.weights
    keep = cp.nonzero(weights != 0)[0]
    coords, weights = coords[keep], weights[keep]
    N = coords.shape[0]
    mo_c = mf.mo_coeff
    mo_o = mf.mo_occ
    nspin = 1 if mult == 1 else 2
    out = np.empty((nspin, 5, N))
    for i0 in range(0, N, block):
        c = coords[i0:i0 + block]
        ao = gni.eval_ao(mol, c, deriv=1, transpose=False)   # (4, nao, blk)
        if nspin == 1:
            r = gni.eval_rho2(mol, ao, mo_c, mo_o, xctype='MGGA')  # total density
            out[0, :, i0:i0 + block] = cp.asnumpy(r) * 0.5         # store per-spin (alpha == beta)
        else:
            for s in range(2):
                r = gni.eval_rho2(mol, ao, mo_c[s], mo_o[s], xctype='MGGA')
                out[s, :, i0:i0 + block] = cp.asnumpy(r)
        del ao
    return out, cp.asnumpy(weights), cp.asnumpy(coords).astype(np.float32)


def libxc_check(rho, weights, mult):
    from pyscf.dft import libxc
    if mult == 1:
        rt = rho[0] * 2
        exc = libxc.eval_xc('wb97m-v', rt, spin=0, deriv=0)[0]
        return float(np.sum(weights * rt[0] * exc))
    exc = libxc.eval_xc('wb97m-v', (rho[0], rho[1]), spin=1, deriv=0)[0]
    return float(np.sum(weights * (rho[0, 0] + rho[1, 0]) * exc))


def process(name, log):
    from pyscf import gto
    import cupy as cp
    t0 = time.time()
    atom, charge, mult = read_xyz(resolve_geometry(name))
    mol = gto.M(atom=atom, basis=BASIS, charge=charge, spin=mult - 1, verbose=0, max_memory=60000)
    mf, attempts, mfx = run_scf(mol, mult, log)
    t_scf = time.time() - t0
    comps = exact_components(mol, mf, mfx, mult)
    t_exact = time.time() - t0 - t_scf
    rho, w, coords = grid_features(mol, mf, mult)
    e_sl_grid = libxc_check(rho, w, mult)
    tmp = os.path.join(DENSITY_DIR, name + '.h5.tmp')
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('rho', data=rho, compression='lzf')
        f.create_dataset('weights', data=w, compression='lzf')
        f.create_dataset('coords', data=coords, compression='lzf')
        mo_c = cp.asnumpy(mf.mo_coeff)
        f.create_dataset('mo_coeff', data=mo_c, compression='lzf')
        f.create_dataset('mo_occ', data=cp.asnumpy(mf.mo_occ))
        f.create_dataset('mo_energy', data=cp.asnumpy(mf.mo_energy))
        for k, v in comps.items():
            f.attrs[k] = v
        f.attrs['E_scf_df'] = float(mf.e_tot)
        f.attrs['E_sl_grid_libxc'] = e_sl_grid
        f.attrs['converged'] = bool(mf.converged)
        f.attrs['attempts'] = json.dumps(attempts)
        f.attrs['charge'] = charge
        f.attrs['mult'] = mult
        f.attrs['natm'] = mol.natm
        f.attrs['nao'] = mol.nao
        f.attrs['nelec'] = mol.nelectron
        f.attrs['ngrid'] = rho.shape[-1]
        f.attrs['basis'] = BASIS
        f.attrs['grid_level'] = GRID_LEVEL
        f.attrs['xc'] = 'wb97m-v'
        f.attrs['protocol'] = 'RIJK(def2-universal-jkfit) SCF + exact-integral energy components'
        f.attrs['t_scf'] = t_scf
        f.attrs['t_exact'] = t_exact
        f.attrs['t_total'] = time.time() - t0
        f.attrs['geometry'] = atom
    os.replace(tmp, os.path.join(DENSITY_DIR, name + '.h5'))
    return dict(name=name, nao=mol.nao, ngrid=int(rho.shape[-1]), converged=bool(mf.converged),
                attempts=attempts, E_tot=comps['E_tot'], dE_sl=e_sl_grid - comps['E_sl'],
                t_scf=round(t_scf, 1), t_exact=round(t_exact, 1), t_total=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='train,validation,test',
                    help='priority order of splits (molecules of earlier splits are computed first)')
    ap.add_argument('--names', nargs='*', help='explicit molecule names')
    ap.add_argument('--order', default='size', choices=['size', 'size_desc', 'file'])
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(DENSITY_DIR, exist_ok=True)
    lockdir = os.path.join(DENSITY_DIR, '_locks')
    os.makedirs(lockdir, exist_ok=True)
    logpath = os.path.join(DENSITY_DIR, '_logs', f'{socket.gethostname()}_gpu{os.environ.get("CUDA_VISIBLE_DEVICES", "x")}.jsonl')
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    if args.names:
        queue = args.names
    else:
        R = load_reactions()
        queue, seen = [], set()
        for sp in args.splits.split(','):
            ms = molecules_for([r for r in R if r.split == sp])
            if args.order != 'file':
                sizes = {}
                for m in ms:
                    atom, _, _ = read_xyz(resolve_geometry(m))
                    sizes[m] = len(atom.split('\n'))
                ms = sorted(ms, key=lambda m: sizes[m], reverse=(args.order == 'size_desc'))
            for m in ms:
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
    n_done = 0
    for name in queue:
        if args.limit and n_done >= args.limit:
            break
        if is_done(name) or not claim(name, lockdir):
            continue
        try:
            res = process(name, log)
            res['status'] = 'ok'
        except Exception as e:  # keep going on failures; they are logged for a retry pass
            res = dict(name=name, status='error', error=repr(e), tb=traceback.format_exc()[-2000:])
        finally:
            try:
                os.remove(os.path.join(lockdir, name + '.lock'))
            except FileNotFoundError:
                pass
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        res['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(logpath, 'a') as f:
            f.write(json.dumps(res) + '\n')
        log(json.dumps(res))
        n_done += 1


if __name__ == '__main__':
    main()
