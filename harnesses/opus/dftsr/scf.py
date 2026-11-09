"""Self-consistent evaluation of a candidate functional with GPU4PySCF.

The candidate's semilocal energy density (PyTorch) replaces libxc inside GPU4PySCF's NumInt:
exc = e / rho and vxc = d e / d(rho, grad rho, tau) via autograd. Exact exchange (ω=0.3, 0.15 SR / 1.0 LR for
the 'rsh' family) and VV10 (b=6, C=0.01) are configured through the libxc name of the frame functional
('wb97m-v' for 'rsh', 'b97m-v' for 'mgga'), exactly as in the reference density calculations.
"""
from __future__ import annotations

import cupy as cp
import torch
from gpu4pyscf.dft import numint as gnumint

from .api import Functional, params_to_tensors
from . import xc_torch as X

FRAME_XC = {'rsh': 'wb97m-v', 'mgga': 'b97m-v'}
BLOCK = 1 << 20


def _density_from_raw(r, spin):
    if spin == 0:   # total-density variables (5, N): rho, grad(3), tau
        ra = 0.5 * r[0]
        g = 0.5 * r[1:4]
        s = (g * g).sum(0)
        t = 0.5 * r[4]
        return X.Density(ra, ra, s, s, s, t, t, closed=True), r[0]
    a, b = r[0], r[1]
    d = X.Density(a[0], b[0], (a[1:4] ** 2).sum(0), (a[1:4] * b[1:4]).sum(0), (b[1:4] ** 2).sum(0), a[4], b[4])
    return d, a[0] + b[0]


class TorchNumInt(gnumint.NumInt):
    """GPU4PySCF NumInt whose semilocal XC comes from a dftsr Functional."""

    def __init__(self, func: Functional, params: dict):
        super().__init__()
        self.func = func
        self.p = params_to_tensors(params, 'cuda')

    def eval_xc_eff(self, xc_code, rho, deriv=1, *, omega=None, xctype=None, spin=None, work=None):
        if deriv > 1:
            raise NotImplementedError('only energies/potentials (deriv<=1) are implemented for custom functionals')
        rho = cp.ascontiguousarray(rho, dtype=cp.float64)
        if spin is None:
            spin = 1 if (rho.ndim == 3 and rho.shape[0] == 2) else 0
        N = rho.shape[-1]
        exc = cp.empty(N)
        vxc = cp.empty_like(rho)
        rt = torch.from_dlpack(rho)
        for i0 in range(0, N, BLOCK):
            r = rt[..., i0:i0 + BLOCK].clone().requires_grad_(True)
            d, rho_tot = _density_from_raw(r, spin)
            e = self.func.energy_density(self.p, d)
            (g,) = torch.autograd.grad(e.sum(), r)
            ex = torch.where(rho_tot > 1e-14, e / rho_tot.clamp_min(1e-30), torch.zeros_like(e))
            exc[i0:i0 + BLOCK] = cp.from_dlpack(ex.detach().contiguous())
            vxc[..., i0:i0 + BLOCK] = cp.from_dlpack(torch.nan_to_num(g).contiguous())
        return exc, vxc, None, None


def make_ks(mol, func: Functional, params: dict, density_fit=True, grid_level=3):
    from gpu4pyscf import dft
    from pyscf.dft import gen_grid
    K = dft.RKS if mol.spin == 0 else dft.UKS
    mf = K(mol, xc=FRAME_XC[func.family])
    if density_fit:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf._numint = TorchNumInt(func, params)
    mf.grids.level = grid_level
    if max(mol.atom_charges()) <= 18:
        mf.nlcgrids.prune = gen_grid.sg1_prune
    return mf
