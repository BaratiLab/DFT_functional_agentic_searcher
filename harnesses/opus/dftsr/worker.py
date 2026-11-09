"""Candidate evaluation pipeline + persistent GPU worker process.

evaluate(path, ...) does:  load -> smoke test -> fit on train (L-BFGS) -> float64 metrics on train/validation
(optionally test) -> constraint suite -> score.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback

import torch

from .api import load_candidate, params_to_tensors, Functional
from .dataset import load_reactions
from .features import SplitData
from .evaluator import Evaluator, ReactionSet, combined_wrmsd
from .constraints import ConstraintSuite
from . import xc_torch as X


def smoke_test(func: Functional, device='cpu'):
    """Evaluate on a few thousand random closed/open-shell points (float64); raise on any problem."""
    g = torch.Generator().manual_seed(0)
    n = 4096
    out = {}
    p = params_to_tensors(func.init_params(), device)
    for closed in (True, False):
        ra = 10 ** (torch.rand(n, generator=g, dtype=torch.float64) * 8 - 6)
        rb = ra if closed else ra * torch.rand(n, generator=g, dtype=torch.float64)
        rb = rb.clone()
        if not closed:
            rb[:16] = 0.0   # fully polarized points (e.g. H atom)
        xa = torch.rand(n, generator=g, dtype=torch.float64) * 6
        xb = xa if closed else torch.rand(n, generator=g, dtype=torch.float64) * 6
        saa = xa ** 2 * ra ** (8 / 3)
        sbb = xb ** 2 * rb ** (8 / 3)
        sab = torch.sqrt(saa * sbb) * (1 if closed else 0.5)
        ta = X.C_TAU * ra ** (5 / 3) * (0.2 + 3 * torch.rand(n, generator=g, dtype=torch.float64)) + saa / (8 * ra)
        tb = ta if closed else X.C_TAU * rb ** (5 / 3) * (0.2 + 3 * torch.rand(n, generator=g, dtype=torch.float64)) \
            + sbb / (8 * rb.clamp_min(1e-30))
        d = X.Density(ra, rb, saa, sab, sbb, ta, tb, closed=closed)
        e = func.energy_density(p, d)
        if e.shape != ra.shape:
            raise ValueError(f'energy_density must return shape [N]={tuple(ra.shape)}, got {tuple(e.shape)}')
        if not torch.isfinite(e).all():
            raise ValueError(f'non-finite energy density on {int((~torch.isfinite(e)).sum())}/{n} random '
                             f'{"closed" if closed else "open"}-shell points (check rho->0, fully polarized points)')
        out['closed' if closed else 'open'] = float(e.sum())
    return out


class Workspace:
    """GPU-resident data for one worker."""

    def __init__(self, device, splits=('train', 'validation'), split_variant='syfes', log=print):
        self.device = device
        self.data = {}
        self.rs = {}
        self.split_variant = split_variant
        self.R = load_reactions(split_variant)
        self.log = log
        for sp in splits:
            self.load_split(sp)
        self.constraints = ConstraintSuite(device)

    def load_split(self, sp):
        if sp in self.data:
            return
        t = time.time()
        self.data[sp] = SplitData(sp, self.device)
        self.rs[sp] = ReactionSet([r for r in self.R if r.split == sp], self.data[sp])
        self.log(f'loaded {sp}: {self.data[sp].nmol} molecules, {self.data[sp].n_points / 1e6:.1f}M points, '
                 f'{len(self.rs[sp].reactions)} reactions ({self.rs[sp].n_missing} missing) in {time.time() - t:.1f}s')


def evaluate(path, ws: Workspace, fit=True, max_iter=300, n_restarts=1, with_test=False, compile_=True,
             j_target=None, time_limit=1800, params_override=None, log=print):
    t0 = time.time()
    res = dict(status='error')
    T = {}                                   # per-stage wall times (s) of this non-SCF calculation

    def lap(key, t_start):
        torch.cuda.synchronize()
        T[key] = round(time.time() - t_start, 2)
        return time.time()

    try:
        t = time.time()
        func = load_candidate(path)
        res['name'] = type(func).__name__
        res['family'] = func.family
        smoke_test(func)
        p0 = params_to_tensors(params_override or func.init_params(), ws.device)
        res['n_params'] = int(sum(v.numel() for v in p0.values()))
        res['n_const'] = len(getattr(func, 'const', {}) or {})
        t = lap('load_and_smoke_test', t)
        ev = Evaluator(func, ws.data, compile_=compile_)
        ev.total_energies(p0, ws.data['train'])          # first pass: triggers torch.compile
        t = lap('compile_first_pass', t)
        if fit:
            p, info = ev.fit(ws.rs['train'], ws.data['train'], p0, max_iter=max_iter, n_restarts=n_restarts,
                             log=log, time_limit=time_limit)
            res['fit'] = info
        else:
            p = p0
        t_fit_end = time.time()
        T['fit'] = round(t_fit_end - t, 2)
        if fit and res['fit'].get('nfev'):
            T['fit_per_feval'] = round(T['fit'] / res['fit']['nfev'], 3)
        t = t_fit_end
        res['compiled'] = ev.compiled
        if ev.compile_error:
            res['compile_fallback'] = ev.compile_error
        res['params'] = {k: [float(x) for x in v.detach().cpu().reshape(-1)] for k, v in p.items()}
        m_tr, _ = ev.metrics(ws.rs['train'], ws.data['train'], p)
        t = lap('metrics_train_fp64', t)
        m_va, _ = ev.metrics(ws.rs['validation'], ws.data['validation'], p)
        t = lap('metrics_validation_fp64', t)
        res['train'] = m_tr
        res['validation'] = m_va
        res['train_wrmsd'] = m_tr['wrmsd']
        res['val_wrmsd'] = m_va['wrmsd']
        if with_test:
            ws.load_split('test')
            m_te, _ = ev.metrics(ws.rs['test'], ws.data['test'], p)
            res['test'] = m_te
            res['test_wrmsd'] = m_te['wrmsd']
            t = lap('metrics_test_fp64', t)
            # "WRMSD_tot" of Duston et al.: validation + test, excluding RG10
            def strip_rg(m):
                bt = {k: v for k, v in m['by_type'].items() if k != 'RG10'}
                n = sum(v['n'] for v in bt.values())
                return dict(n=n, wrmsd=math.sqrt(sum(v['n'] * v['wrmsd'] ** 2 for v in bt.values()) / n))
            res['valtest_noRG10_wrmsd'] = combined_wrmsd(strip_rg(m_va), strip_rg(m_te))
        nonfinite = m_tr['nonfinite_points'] + m_va['nonfinite_points']
        if ws.constraints.ok:
            c = ws.constraints.run(func, p)
            t = lap('constraints', t)
        else:
            c = dict(n_violations=0, note='constraint data missing')
        if nonfinite:
            c['numerical'] = dict(value=nonfinite, passed=False, note='non-finite energy density on dataset points')
            c['n_violations'] += 1
        res['constraints'] = c
        res['n_violations'] = c['n_violations']
        if j_target and math.isfinite(res['val_wrmsd']):
            res['score'] = j_target / res['val_wrmsd'] * 0.9 ** c['n_violations']
        res['status'] = 'ok' if math.isfinite(res['val_wrmsd']) else 'error'
        if res['status'] == 'error':
            res['error'] = 'validation WRMSD is not finite'
    except Exception as e:
        res['error'] = f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()[-3000:]
    res['eval_seconds'] = round(time.time() - t0, 1)
    T['total_gpu_eval'] = res['eval_seconds']
    res['timings'] = T
    return res


def worker_main(gpu: int, conn, splits=('train', 'validation'), split_variant='syfes'):
    """Persistent worker loop (multiprocessing target). Receives job dicts, sends result dicts."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)   # before any CUDA call in this process
    torch.set_num_threads(4)
    logs = []

    def log(m):
        logs.append(m)

    try:
        ws = Workspace('cuda:0', splits, split_variant, log=log)
        conn.send(dict(ready=True, log=logs[:]))
    except Exception as e:
        conn.send(dict(ready=False, error=repr(e), traceback=traceback.format_exc()))
        return
    while True:
        job = conn.recv()
        if job is None:
            break
        logs.clear()
        opts = job.get('opts', {})
        res = evaluate(job['path'], ws, log=log, **opts)
        res['log'] = logs[-50:]
        res['gpu'] = gpu
        conn.send(res)
        torch.cuda.empty_cache()


if __name__ == '__main__':  # CLI: python -m dftsr.worker candidate.py [--test] [--nofit]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--nofit', action='store_true')
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--no-compile', action='store_true')
    a = ap.parse_args()
    ws = Workspace('cuda:0')
    r = evaluate(a.path, ws, fit=not a.nofit, with_test=a.test, n_restarts=a.restarts, max_iter=a.max_iter,
                 compile_=not a.no_compile)
    r.pop('traceback', None) if r['status'] == 'ok' else None
    print(json.dumps({k: v for k, v in r.items() if k not in ('train', 'validation', 'test')}, indent=1))
    for sp in ('validation', 'test'):
        if sp in r:
            print(sp, json.dumps(r[sp]['by_type'], indent=0))
