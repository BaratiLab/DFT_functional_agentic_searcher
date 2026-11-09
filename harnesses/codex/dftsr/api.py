"""Candidate-functional API.

A candidate is a Python file defining exactly one subclass of `Functional`. Template:

    from dftsr.api import *

    class MyFunctional(Functional):
        family = 'rsh'                       # 'rsh': ωB97M-V frame (0.15 SR-HF + 1.0 LR-HF, ω=0.3, VV10 fixed)
                                             # 'mgga': no exact exchange (B97M-V frame, VV10 fixed)
        def init_params(self):
            return {'cx': [0.259, 1.007], ...}          # ALL trainable parameters, refit from these values

        def exchange(self, p, d):                       # semilocal (short-range for 'rsh') exchange energy density
            def fx(ch):                                 # ch: SpinChannel (per-spin rho, sigma, tau, x2, t, w, ...)
                return 0.85 + p['cx'][0] * ch.w + p['cx'][1] * b97_u(ch.x2, 0.004)
            return exchange_from_fx(d, fx, self.omega)

        def correlation(self, p, d):                    # semilocal correlation energy density
            return correlation_from_factors(d, fss=lambda ch: ..., fos=lambda d: ...)

Both methods return a tensor of shape [N] (energy per unit volume, Hartree/bohr^3), summed over spins.
`p` maps parameter name -> torch tensor (1-D). `self.const` holds fixed (non-trained) constants.
"""
from __future__ import annotations

import importlib.util
import inspect
import math

import torch

from .xc_torch import (  # noqa: F401  (re-exported for candidate files)
    PI, C_TAU, C_X_LDA, RHO_THR, Density, SpinChannel, WB97MV, B97MV,
    b97_u, b97_w, power_series, f_rsh, f_zeta, e_x_lda_spin, e_c_pw92, e_c_pw92_fully_polarized,
    eps_c_pw92, pw92_g, wigner_seitz_radius,
)

FAMILIES = {
    # exact-exchange fractions: E_x^HF contribution = cx_sr * E_xx^SR + cx_lr * E_xx^LR (ω = 0.3)
    'rsh': dict(omega=0.3, cx_sr=0.15, cx_lr=1.0, ueg_gx=0.85),
    'mgga': dict(omega=0.0, cx_sr=0.0, cx_lr=0.0, ueg_gx=1.0),
}

softplus = torch.nn.functional.softplus
sigmoid = torch.sigmoid
tanh = torch.tanh


class Functional:
    """Base class. Subclasses implement init_params(), exchange(p, d), correlation(p, d)."""
    family: str = 'rsh'
    const: dict = {}

    @property
    def omega(self):
        return FAMILIES[self.family]['omega']

    def init_params(self) -> dict:
        raise NotImplementedError

    def exchange(self, p: dict, d: Density) -> torch.Tensor:
        raise NotImplementedError

    def correlation(self, p: dict, d: Density) -> torch.Tensor:
        raise NotImplementedError

    def energy_density(self, p: dict, d: Density) -> torch.Tensor:
        return self.exchange(p, d) + self.correlation(p, d)


# ---------------------------------------------------------------------------- helpers
def spin_sum(d: Density, fn):
    """fn(channel) -> tensor; returns fn(a) + fn(b), zeroing empty channels. Uses 2*fn(a) for closed shells."""
    if getattr(d, 'closed', False):
        a = d.a
        return 2 * torch.where(a.mask, fn(a), 0.0)
    return sum(torch.where(ch.mask, fn(ch), 0.0) for ch in (d.a, d.b))


def exchange_from_fx(d: Density, fx, omega: float):
    """sum_sigma e_x^LDA(rho_sigma) * F_rsh(rho_sigma; omega) * fx(channel)."""
    return spin_sum(d, lambda ch: ch.e_x_lda * f_rsh(ch.rho, omega) * fx(ch))


def correlation_from_factors(d: Density, fss, fos):
    """sum_sigma e_c,ss^LDA(sigma) fss(channel) + e_c,os^LDA fos(d)  (Stoll partition of PW92)."""
    eaa, ebb, eab = d.e_c_lda_decomposed()
    if getattr(d, 'closed', False):
        ess = 2 * torch.where(d.a.mask, eaa * fss(d.a), 0.0)
    else:
        ess = torch.where(d.a.mask, eaa * fss(d.a), 0.0) + torch.where(d.b.mask, ebb * fss(d.b), 0.0)
    return ess + eab * fos(d)


def load_candidate(path: str) -> Functional:
    """Import a candidate file and return an instance of its single Functional subclass."""
    spec = importlib.util.spec_from_file_location('candidate_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if issubclass(c, Functional) and c is not Functional and c.__module__ == mod.__name__]
    if len(classes) != 1:
        raise ValueError(f'candidate file must define exactly one Functional subclass, found {len(classes)}')
    obj = classes[0]()
    if obj.family not in FAMILIES:
        raise ValueError(f'family must be one of {list(FAMILIES)}')
    return obj


def params_to_tensors(params: dict, device, dtype=torch.float64):
    out = {}
    for k, v in params.items():
        t = torch.as_tensor(v, dtype=dtype, device=device)
        out[k] = t.reshape(-1) if t.ndim == 0 else t
    return out
