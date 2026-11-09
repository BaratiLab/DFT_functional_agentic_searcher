"""Non-SCF evaluation of a candidate functional on fixed ωB97M-V densities.

E_mol(candidate) = E_tot^ωB97M-V
                 + [E_HF-x(candidate family) - E_HF-x(ωB97M-V)]
                 + E_sl^candidate[rho] - E_sl^ωB97M-V[rho]
with E_HF-x(family) = cx_sr (E_xx - E_xx_lr) + cx_lr E_xx_lr.  VV10 (b=6, C=0.01) is kept fixed.

Parameters are fitted on the training split by minimising WRMSD_train^2 with L-BFGS (PyTorch autograd,
float32 compiled kernels); reported metrics are recomputed in float64.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch

from .api import FAMILIES, Functional, params_to_tensors
from .dataset import HARTREE_TO_KCAL, load_reactions
from .features import SplitData
from . import xc_torch as X

CHUNK = 1 << 22


class FitTimeout(Exception):
    """Raised inside the L-BFGS closure when the fitting time budget is exhausted."""


class ReactionSet:
    """Stoichiometry of a list of reactions against the molecules of a SplitData."""

    def __init__(self, reactions, data: SplitData):
        rows, cols, vals, keep = [], [], [], []
        for r in reactions:
            if all(m in data.index for _, m in r.stoich):
                i = len(keep)
                keep.append(r)
                for c, m in r.stoich:
                    rows.append(i)
                    cols.append(data.index[m])
                    vals.append(float(c))
        self.reactions = keep
        self.n_missing = len(reactions) - len(keep)
        dev = data.device
        self.S = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float64),
                                         (len(keep), data.nmol)).coalesce().to(dev)
        self.ref = torch.tensor([r.ref * HARTREE_TO_KCAL for r in keep], dtype=torch.float64, device=dev)
        self.w = torch.tensor([r.weight for r in keep], dtype=torch.float64, device=dev)

    def predict(self, E):  # E: per-molecule total energies (Hartree) -> reaction energies (kcal/mol)
        return torch.sparse.mm(self.S, E.unsqueeze(1)).squeeze(1) * HARTREE_TO_KCAL

    def wrmsd(self, E):
        return torch.sqrt(torch.mean(self.w * (self.predict(E) - self.ref) ** 2))


def _make_kernels(func: Functional, compile_: bool):
    def closed_k(p, rho, sigma, tau):
        d = X.Density(rho, rho, sigma, sigma, sigma, tau, tau, closed=True)
        return func.energy_density(p, d)

    def open_k(p, ra, rb, saa, sab, sbb, ta, tb):
        d = X.Density(ra, rb, saa, sab, sbb, ta, tb)
        return func.energy_density(p, d)

    if compile_:
        closed_k = torch.compile(closed_k, dynamic=False)
        open_k = torch.compile(open_k, dynamic=False)
    return closed_k, open_k


class Evaluator:
    def __init__(self, func: Functional, data: dict, compile_=True):
        self.func = func
        self.data = data          # split -> SplitData
        self.fam = FAMILIES[func.family]
        self.kernels = {torch.float32: _make_kernels(func, compile_), torch.float64: _make_kernels(func, False)}
        self.compiled = compile_
        self.compile_error = None

    def _call(self, dtype, which, *args):
        try:
            return self.kernels[dtype][which](*args)
        except Exception as e:
            if dtype == torch.float32 and self.compiled:
                # torch.compile could not handle the candidate code: fall back to eager mode
                self.compile_error = f'{type(e).__name__}: {str(e)[:300]}'
                self.compiled = False
                self.kernels[torch.float32] = _make_kernels(self.func, False)
                return self.kernels[dtype][which](*args)
            raise

    # ------------------------------------------------------------------ energies
    def _chunks(self, block, keys, n):
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            sl = [block[k][i0:i1] for k in keys]
            pad = CHUNK - (i1 - i0)
            if pad:   # pad to a fixed shape for the compiled kernel (zero weight)
                sl = [torch.cat([x, x[:1].expand(pad)]) for x in sl]
                sl[-1] = sl[-1].clone()
                sl[-1][i1 - i0:] = 0
            yield i0, i1, sl

    def e_sl(self, p, data: SplitData, dtype=torch.float32, grad_E=None, check_finite=False):
        """Per-molecule semilocal XC energies. If grad_E is given, backprop sum(grad_E * E_sl) into p."""
        E = torch.zeros(data.nmol, dtype=torch.float64, device=data.device)
        pc = {k: v.to(dtype) for k, v in p.items()}
        bad = 0
        for kind, keys, kern in (('closed', ('rho', 'sigma', 'tau', 'w'), 0),
                                 ('open', ('ra', 'rb', 'saa', 'sab', 'sbb', 'ta', 'tb', 'w'), 1)):
            block = getattr(data, kind)
            n = block['w'].numel()
            if n == 0:
                continue
            mol = block['mol']
            for i0, i1, sl in self._chunks(block, keys, n):
                feats = [x.to(dtype) for x in sl[:-1]]
                w = sl[-1].to(dtype)
                with torch.set_grad_enabled(grad_E is not None):
                    e = self._call(dtype, kern, pc, *feats)
                    contrib = (w * e)[: i1 - i0].double()
                    if check_finite:
                        bad += int((~torch.isfinite(contrib)).sum())
                    if grad_E is not None:
                        (grad_E[mol[i0:i1]] * contrib).sum().backward()
                E.index_add_(0, mol[i0:i1], contrib.detach())
        if check_finite:
            return E, bad
        return E

    def total_energies(self, p, data: SplitData, dtype=torch.float32, **kw):
        out = self.e_sl(p, data, dtype, **kw)
        E_sl, extra = (out if isinstance(out, tuple) else (out, None))
        sc = data.scalars
        hf_ref = 0.15 * (sc['E_xx'] - sc['E_xx_lr']) + 1.0 * sc['E_xx_lr']
        hf = self.fam['cx_sr'] * (sc['E_xx'] - sc['E_xx_lr']) + self.fam['cx_lr'] * sc['E_xx_lr']
        E = sc['E_tot'] + (hf - hf_ref) + (E_sl - sc['E_sl_t64'])
        return (E, extra) if extra is not None else E

    # ------------------------------------------------------------------ fitting
    def fit(self, rs: ReactionSet, data: SplitData, p0: dict, max_iter=300, n_restarts=1, seed=0, log=None,
            time_limit=None):
        names = list(p0)
        shapes = [p0[k].numel() for k in names]
        dev = data.device

        def unflat(theta):
            out, i = {}, 0
            for k, n in zip(names, shapes):
                out[k] = theta[i:i + n]
                i += n
            return out

        flat0 = torch.cat([p0[k].reshape(-1) for k in names]).double().to(dev)
        nfev = [0]
        seen = {'best': None}          # best (loss, theta) evaluated so far, across restarts
        t_start = time.time()

        def loss_and_grad(theta):
            if time_limit and time.time() - t_start > time_limit and seen['best'] is not None:
                raise FitTimeout()
            nfev[0] += 1
            p = unflat(theta.detach())
            E = self.total_energies(p, data).requires_grad_(True)
            L = rs.wrmsd(E) ** 2
            if not torch.isfinite(L):
                return torch.tensor(1e8, dtype=torch.float64, device=dev), torch.zeros_like(theta)
            (gE,) = torch.autograd.grad(L, E)
            pp = {k: v.clone().requires_grad_(True) for k, v in p.items()}
            self.e_sl(pp, data, grad_E=gE)
            g = torch.cat([(pp[k].grad if pp[k].grad is not None else torch.zeros_like(pp[k])).reshape(-1)
                           for k in names])
            if not torch.isfinite(g).all():
                return torch.tensor(1e8, dtype=torch.float64, device=dev), torch.zeros_like(theta)
            if seen['best'] is None or float(L) < seen['best'][0]:
                seen['best'] = (float(L), theta.detach().clone())
            return L.detach(), g

        best = None
        gen = torch.Generator(device='cpu').manual_seed(seed)
        history = []
        timed_out = False
        for k in range(max(1, n_restarts)):
            theta = flat0.clone()
            if k > 0:
                theta = theta + (0.1 * (theta.abs() + 0.1)) * torch.randn(theta.shape, generator=gen).double().to(dev)
            theta.requires_grad_(True)
            opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, max_eval=int(max_iter * 1.5),
                                    tolerance_grad=1e-10, tolerance_change=1e-12, history_size=30,
                                    line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                L, g = loss_and_grad(theta)
                theta.grad = g
                return L

            try:
                opt.step(closure)
                L, _ = loss_and_grad(theta)
                L = float(L)
            except FitTimeout:
                timed_out = True
                L, th = seen['best']
                theta = th.clone()
                if log:
                    log(f'restart {k}: fit time budget ({time_limit:.0f}s) exhausted; keeping best parameters so far')
            except Exception as e:  # numerical blow-up inside line search etc.
                L = float('inf')
                if log:
                    log(f'restart {k}: optimizer error {e!r}')
            history.append(math.sqrt(L) if L < 1e7 else None)
            if log:
                log(f'restart {k}: train WRMSD {history[-1]} (nfev so far {nfev[0]})')
            if best is None or L < best[0]:
                best = (L, theta.detach().clone())
            if timed_out or (time_limit and time.time() - t_start > time_limit):
                break
        if seen['best'] is not None and (best is None or seen['best'][0] < best[0]):
            best = seen['best']
        return unflat(best[1]), dict(nfev=nfev[0], restarts=history, timed_out=timed_out,
                                     fit_seconds=round(time.time() - t_start, 1),
                                     train_wrmsd_fp32=min([h for h in history if h is not None], default=None))

    # ------------------------------------------------------------------ metrics
    def metrics(self, rs: ReactionSet, data: SplitData, p, dtype=torch.float64):
        E, bad = self.total_energies(p, data, dtype=dtype, check_finite=True)
        pred = rs.predict(E)
        err = pred - rs.ref
        out = dict(wrmsd=float(torch.sqrt(torch.mean(rs.w * err ** 2))), n=len(rs.reactions),
                   n_missing=rs.n_missing, nonfinite_points=bad)
        err_np = err.cpu().numpy()
        w_np = rs.w.cpu().numpy()
        by_t, by_s = defaultdict(list), defaultdict(list)
        for i, r in enumerate(rs.reactions):
            by_t[r.dtype].append(i)
            by_s[r.subset].append(i)
        out['by_type'] = {t: dict(wrmsd=float(np.sqrt(np.mean(w_np[ix] * err_np[ix] ** 2))),
                                  rmsd=float(np.sqrt(np.mean(err_np[ix] ** 2))), n=len(ix))
                          for t, ix in sorted(by_t.items())}
        out['by_subset_rmsd'] = {s: round(float(np.sqrt(np.mean(err_np[ix] ** 2))), 4)
                                 for s, ix in sorted(by_s.items())}
        return out, E


def combined_wrmsd(m1, m2):
    """WRMSD over the union of two disjoint reaction sets."""
    n1, n2 = m1['n'], m2['n']
    return math.sqrt((n1 * m1['wrmsd'] ** 2 + n2 * m2['wrmsd'] ** 2) / (n1 + n2))
