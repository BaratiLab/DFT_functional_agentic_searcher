"""Build / load compact per-split feature caches from the per-molecule density HDF5 files.

Cache layout (FEATURE_CACHE_DIR/<split>/):
  closed_{rho,sigma,tau,w}.npy  float32 per-spin quantities of closed-shell molecules (rho_a = rho_b)
  closed_mol.npy                int32   molecule index of each point
  open_{ra,rb,saa,sab,sbb,ta,tb,w}.npy, open_mol.npy  for open-shell molecules
  mols.json                     molecule names (index order) + per-molecule scalars
  scalars.npz                   float64 arrays: E_tot, E_sl (libxc), E_sl_t64 (torch wB97M-V, float64),
                                E_xx, E_xx_lr, E_nlc, converged
Grid points with total density < DROP_RHO are discarded (their wB97M-V contribution is recorded and is
negligible, < 1e-9 Ha per molecule).
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch

from .config import DENSITY_DIR, FEATURE_CACHE_DIR
from .dataset import load_reactions, molecules_for
from . import xc_torch as X

DROP_RHO = 1e-11
CLOSED_KEYS = ('rho', 'sigma', 'tau', 'w')
OPEN_KEYS = ('ra', 'rb', 'saa', 'sab', 'sbb', 'ta', 'tb', 'w')


def density_path(name):
    return os.path.join(DENSITY_DIR, name + '.h5')


def available(name):
    p = density_path(name)
    if not os.path.exists(p):
        return False
    try:
        with h5py.File(p, 'r') as f:
            return 'E_tot' in f.attrs
    except OSError:
        return False


def read_molecule(name):
    """Return (kind, dict of float64 arrays, attrs) for one molecule."""
    with h5py.File(density_path(name), 'r') as f:
        rho = f['rho'][:]
        w = f['weights'][:]
        attrs = {k: f.attrs[k] for k in ('E_tot', 'E_sl', 'E_xx', 'E_xx_lr', 'E_nlc', 'converged', 'mult')}
    if rho.shape[0] == 1:
        a = rho[0]
        keep = 2 * a[0] > DROP_RHO
        a = a[:, keep]
        d = dict(rho=a[0], sigma=(a[1:4] ** 2).sum(0), tau=a[4], w=w[keep])
        return 'closed', d, attrs, keep
    a, b = rho[0], rho[1]
    keep = (a[0] + b[0]) > DROP_RHO
    a, b = a[:, keep], b[:, keep]
    d = dict(ra=a[0], rb=b[0], saa=(a[1:4] ** 2).sum(0), sab=(a[1:4] * b[1:4]).sum(0),
             sbb=(b[1:4] ** 2).sum(0), ta=a[4], tb=b[4], w=w[keep])
    return 'open', d, attrs, keep


def density_from_arrays(kind, d, dtype=torch.float64, device='cpu'):
    T = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    if kind == 'closed':
        r, s, t = T(d['rho']), T(d['sigma']), T(d['tau'])
        return X.Density(r, r, s, s, s, t, t, closed=True)
    return X.Density(T(d['ra']), T(d['rb']), T(d['saa']), T(d['sab']), T(d['sbb']), T(d['ta']), T(d['tb']))


def build_cache(split, names=None, out_dir=None, log=print):
    """Concatenate all available molecules of `split` into a feature cache (float32)."""
    R = load_reactions()
    if names is None:
        names = molecules_for([r for r in R if r.split == split]) if split != 'all' else molecules_for(R)
    names = [n for n in names if available(n)]
    out_dir = out_dir or os.path.join(FEATURE_CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    buf = {'closed': {k: [] for k in CLOSED_KEYS + ('mol',)}, 'open': {k: [] for k in OPEN_KEYS + ('mol',)}}
    sc = {k: [] for k in ('E_tot', 'E_sl', 'E_sl_t64', 'E_xx', 'E_xx_lr', 'E_nlc', 'converged', 'npts')}
    for i, n in enumerate(names):
        kind, d, attrs, _ = read_molecule(n)
        e = X.e_xc_b97m_family(density_from_arrays(kind, d))
        sc['E_sl_t64'].append(float((torch.as_tensor(d['w']) * e).sum()))
        for k in ('E_tot', 'E_sl', 'E_xx', 'E_xx_lr', 'E_nlc', 'converged'):
            sc[k].append(float(attrs[k]))
        sc['npts'].append(len(d['w']))
        for k, v in d.items():
            buf[kind][k].append(v.astype(np.float32))
        buf[kind]['mol'].append(np.full(len(d['w']), i, dtype=np.int32))
        if (i + 1) % 500 == 0:
            log(f'  {split}: {i + 1}/{len(names)} molecules')
    for kind, dd in buf.items():
        for k, v in dd.items():
            arr = np.concatenate(v) if v else np.zeros(0, dtype=np.int32 if k == 'mol' else np.float32)
            np.save(os.path.join(out_dir, f'{kind}_{k}.npy'), arr)
    np.savez(os.path.join(out_dir, 'scalars.npz'), **{k: np.array(v) for k, v in sc.items()})
    json.dump(dict(names=names, n_points=int(sum(sc['npts']))), open(os.path.join(out_dir, 'mols.json'), 'w'))
    log(f'cache {split}: {len(names)} molecules, {sum(sc["npts"]) / 1e6:.1f}M points -> {out_dir}')
    return out_dir


class SplitData:
    """GPU-resident features for one split + per-molecule reference scalars."""

    def __init__(self, split, device, cache_dir=None):
        cache_dir = cache_dir or os.path.join(FEATURE_CACHE_DIR, split)
        self.split = split
        self.device = device
        meta = json.load(open(os.path.join(cache_dir, 'mols.json')))
        self.names = meta['names']
        self.index = {n: i for i, n in enumerate(self.names)}
        s = np.load(os.path.join(cache_dir, 'scalars.npz'))
        self.scalars = {k: torch.as_tensor(s[k], dtype=torch.float64, device=device) for k in s.files}
        self.closed = {k: torch.from_numpy(np.load(os.path.join(cache_dir, f'closed_{k}.npy'))).to(device)
                       for k in CLOSED_KEYS + ('mol',)}
        self.open = {k: torch.from_numpy(np.load(os.path.join(cache_dir, f'open_{k}.npy'))).to(device)
                     for k in OPEN_KEYS + ('mol',)}
        self.closed['mol'] = self.closed['mol'].long()
        self.open['mol'] = self.open['mol'].long()
        self.nmol = len(self.names)
        self.n_points = int(self.closed['w'].numel() + self.open['w'].numel())
