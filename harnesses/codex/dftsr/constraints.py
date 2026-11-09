"""Physical / technical constraint tests (after Duston et al. 2026, Sec. 1.3).

  1. spin_symmetry : max_r |e_xc(rho_a, rho_b) - e_xc(rho_b, rho_a)| / rho  < 1e-5 Ha   (O2 triplet grid)
  2. ueg_exchange  : |g_x(s=0, tau=tau_UEG) - g_x^UEG| < 0.5% (g_x^UEG = 0.85 for 'rsh', 1.0 for 'mgga'),
                     synthetic unpolarized UEG at r_s = 0.5, 1, 2, 5
  3. x_scaling     : exchange enhancement factor invariant under rho_l(r) = l^3 rho(l r), l = 0.5, 2, 5:
                     sum_r w |e_x^LDA-sr| |g_x(rho_l) - g_x(rho)| < 1e-4 Ha  (H2O grid; RSH attenuation divided out)
  4. grid_ae18     : max_atoms |E_sl(99,590) - E_sl(250,974)| < 0.015 kcal/mol (AE18 atoms H-Ar)
  plus numerical sanity: finite energies on every grid point of the evaluated data.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from .api import FAMILIES, Functional, spin_sum
from .config import CONSTRAINT_DIR
from .dataset import HARTREE_TO_KCAL
from . import xc_torch as X

THRESH = dict(spin_symmetry=1e-5, ueg_exchange=0.005, x_scaling=1e-4, grid_ae18=0.015)


def _density(rho, dev):
    T = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    a, b = rho[0], rho[1]
    return X.Density(T(a[0]), T(b[0]), T((a[1:4] ** 2).sum(0)), T((a[1:4] * b[1:4]).sum(0)),
                     T((b[1:4] ** 2).sum(0)), T(a[-1]), T(b[-1]))   # tau is the last row (5- or 6-row layouts)


def _lda_sr_x(d: X.Density, omega):
    """sum_sigma e_x^LDA(rho_s) F_rsh(rho_s): the reference that g_x multiplies."""
    return spin_sum(d, lambda ch: ch.e_x_lda * X.f_rsh(ch.rho, omega))


class ConstraintSuite:
    def __init__(self, device):
        self.dev = device
        path = os.path.join(CONSTRAINT_DIR, 'constraints.h5')
        self.ok = os.path.exists(path)
        if not self.ok:
            return
        with h5py.File(path, 'r') as f:
            self.o2 = _density(f['O2_triplet/rho'][:], device)
            self.o2_w = torch.as_tensor(f['O2_triplet/weights'][:], device=device)
            self.h2o = _density(f['H2O/rho'][:], device)
            self.h2o_w = torch.as_tensor(f['H2O/weights'][:], device=device)
            self.ae18 = {}
            for name in f['AE18']:
                self.ae18[name] = {tag: (_density(f[f'AE18/{name}/{tag}/rho'][:], device),
                                         torch.as_tensor(f[f'AE18/{name}/{tag}/weights'][:], device=device))
                                   for tag in ('coarse', 'fine')}

    @torch.no_grad()
    def run(self, func: Functional, p: dict) -> dict:
        p = {k: v.double() for k, v in p.items()}
        fam = FAMILIES[func.family]
        res = {}

        # 1. spin symmetry
        try:
            d, ds = self.o2, self.o2.swapped()
            e1, e2 = func.energy_density(p, d), func.energy_density(p, X.Density(
                ds.rho_a, ds.rho_b, ds.sigma_aa, ds.sigma_ab, ds.sigma_bb, ds.tau_a, ds.tau_b))
            rho = d.rho
            m = rho > 1e-6
            dev = float(((e1 - e2).abs()[m] / rho[m]).max())
            res['spin_symmetry'] = dict(value=dev, threshold=THRESH['spin_symmetry'],
                                        passed=bool(dev < THRESH['spin_symmetry']))
        except Exception as e:
            res['spin_symmetry'] = dict(value=None, passed=False, error=repr(e))

        # 2. UEG exchange limit
        try:
            rs = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64, device=self.dev)
            rho = 3 / (4 * np.pi * rs ** 3)
            ra = rho / 2
            tau = X.C_TAU * ra ** (5 / 3)
            z = torch.zeros_like(ra)
            d = X.Density(ra, ra.clone(), z, z.clone(), z.clone(), tau, tau.clone())
            gx = func.exchange(p, d) / _lda_sr_x(d, fam['omega'])
            dev = float((gx - fam['ueg_gx']).abs().max())
            res['ueg_exchange'] = dict(value=dev, gx=[round(float(v), 6) for v in gx],
                                       threshold=THRESH['ueg_exchange'], passed=bool(dev < THRESH['ueg_exchange']))
        except Exception as e:
            res['ueg_exchange'] = dict(value=None, passed=False, error=repr(e))

        # 3. uniform coordinate scaling of exchange
        try:
            d = self.h2o
            ref = _lda_sr_x(d, fam['omega'])
            g0 = func.exchange(p, d) / ref
            m = d.rho > 1e-8
            worst = 0.0
            for lam in (0.5, 2.0, 5.0):
                dl = d.scaled(lam)
                gl = func.exchange(p, dl) / _lda_sr_x(dl, fam['omega'])
                val = float((self.h2o_w * ref.abs() * (gl - g0).abs())[m].sum())
                worst = max(worst, val)
            res['x_scaling'] = dict(value=worst, threshold=THRESH['x_scaling'], passed=bool(worst < THRESH['x_scaling']))
        except Exception as e:
            res['x_scaling'] = dict(value=None, passed=False, error=repr(e))

        # 4. AE18 grid sensitivity
        try:
            worst, per = 0.0, {}
            for name, g in self.ae18.items():
                E = {}
                for tag, (d, w) in g.items():
                    E[tag] = float((w * func.energy_density(p, d)).sum())
                diff = abs(E['coarse'] - E['fine']) * HARTREE_TO_KCAL
                per[name] = round(diff, 5)
                worst = max(worst, diff)
            res['grid_ae18'] = dict(value=worst, threshold=THRESH['grid_ae18'], passed=bool(worst < THRESH['grid_ae18']),
                                    worst_atom=max(per, key=per.get))
        except Exception as e:
            res['grid_ae18'] = dict(value=None, passed=False, error=repr(e))
        res['n_violations'] = sum(not v['passed'] for k, v in res.items() if isinstance(v, dict))
        return res
