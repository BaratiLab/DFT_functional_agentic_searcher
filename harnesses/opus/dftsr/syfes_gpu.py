"""GPU evaluation and CMA-ES training of SyFES forms on the MGCDB84 non-SCF caches (ωB97M-V densities).

Speed design (results identical to evaluating the form point by point):
  * Parameter-independent per-point quantities are precomputed once per split and stored as "rows":
      same-spin rows (one per occupied spin channel): x2_s, w_s, p_x = w_grid e_x^LDA F_rsh, p_ss = w_grid e_c,ss^LDA
      opposite-spin rows (one per point):            x2_avg, w_avg, p_os = w_grid e_c,os^LDA
    (closed shells: one same-spin row with prefactors x2). E_sl[mol] = sum p_x F_x + p_ss F_css + p_os F_cos.
  * Rows are grouped per molecule and padded to blocks of BLK rows, so every block belongs to one molecule; a
    torch.compile'd kernel evaluates the form for a whole batch of parameter vectors [P, k] (all CMA-ES samples of
    all 5 restarts at once) and reduces each block in float32; block sums -> molecules in float64.
  * Total energy as in dftsr/evaluator.py ('rsh' frame): E = E_tot(ωB97M-V) + E_sl(form) - E_sl,t64(ωB97M-V).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import syfes as S
from . import xc_torch as X
from .config import FEATURE_CACHE_DIR, OMEGA
from .dataset import load_reactions
from .evaluator import ReactionSet
from .features import SplitData

BLK = 128
CHUNK = 1 << 20          # rows per kernel call (multiple of BLK)
ROWS_DIR = os.path.join(os.path.dirname(FEATURE_CACHE_DIR), 'syfes_rows', os.path.basename(FEATURE_CACHE_DIR))
# Adaptive precision: float32 accumulation while a restart's last generation spans > SWITCH kcal/mol (noise
# ~4e-5 kcal/mol is irrelevant to the ranking there), float64 accumulation (noise ~5e-6) once any restart is closer.
SUB = int(os.environ.get('SYFES_SUB', '8'))
SWITCH = float(os.environ.get('SYFES_PREC_SWITCH', '0.01'))
BIG = 1e10               # loss assigned to non-finite WRMSD (pycma<=3.0 passed NaN through as the worst value)


# ============================================================================ row data
# Row types: 'cl' = closed-shell points (same-spin and opposite-spin features coincide: x2_avg = x2_a, w_avg = w_a),
# 'oss' = open-shell occupied spin channels, 'oos' = open-shell opposite-spin rows.
ROW_TYPES = {'cl': ('px', 'pss', 'pos'), 'oss': ('px', 'pss'), 'oos': ('pos',)}


def _rows_from_density(d: X.Density, w: torch.Tensor, mol: torch.Tensor, closed: bool):
    eaa, ebb, eab = d.e_c_lda_decomposed()
    out = {t: [] for t in ROW_TYPES}
    if closed:
        ch = d.a
        m = ch.mask
        px = 2 * w * ch.e_x_lda * X.f_rsh(ch.rho, OMEGA)
        pss = 2 * w * eaa
        pos = w * eab
        # masked-out points (rho_s <= RHO_THR) keep their opposite-spin row with px = pss = 0
        out['cl'].append(dict(x2=torch.where(m, ch.x2, d.x2_avg), w=torch.where(m, ch.w, d.w_avg),
                              px=torch.where(m, px, 0.0), pss=torch.where(m, pss, 0.0), pos=pos, mol=mol))
        return out
    for ch, ec in ((d.a, eaa), (d.b, ebb)):
        m = ch.mask
        px = w * ch.e_x_lda * X.f_rsh(ch.rho, OMEGA)
        out['oss'].append(dict(x2=ch.x2[m], w=ch.w[m], px=px[m], pss=(w * ec)[m], mol=mol[m]))
    out['oos'].append(dict(x2=d.x2_avg, w=d.w_avg, pos=w * eab, mol=mol))
    return out


def build_rows(split: str, device='cuda', out_dir=None, log=print):
    """Precompute the row arrays of one split (float64 math, stored float32) and save them as .npy files."""
    out_dir = out_dir or os.path.join(ROWS_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    data = SplitData(split, device)
    acc = {t: {k: [] for k in ('x2', 'w', *pf, 'mol')} for t, pf in ROW_TYPES.items()}
    step = 1 << 22
    for kind in ('closed', 'open'):
        blk = getattr(data, kind)
        n = blk['w'].numel()
        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            g = lambda k: blk[k][i0:i1].double()
            if kind == 'closed':
                r, s_, t = g('rho'), g('sigma'), g('tau')
                d = X.Density(r, r, s_, s_, s_, t, t, closed=True)
            else:
                d = X.Density(g('ra'), g('rb'), g('saa'), g('sab'), g('sbb'), g('ta'), g('tb'))
            for typ, lst in _rows_from_density(d, g('w'), blk['mol'][i0:i1], kind == 'closed').items():
                for r_ in lst:
                    for k, v in r_.items():
                        acc[typ][k].append(v.float().cpu() if k != 'mol' else v.int().cpu())
    nmol = data.nmol
    meta = dict(split=split, nmol=nmol, names=data.names, blk=BLK, chunk=CHUNK, layout=2)
    for typ, dd in acc.items():
        if not dd['mol']:
            dd = {k: [torch.zeros(0, dtype=torch.int32 if k == 'mol' else torch.float32)] for k in dd}
        cols = {k: torch.cat(v) for k, v in dd.items()}
        mol = cols.pop('mol').long()
        order = torch.argsort(mol, stable=True)
        mol = mol[order]
        cols = {k: v[order] for k, v in cols.items()}
        counts = torch.bincount(mol, minlength=nmol)
        padded = ((counts + BLK - 1) // BLK) * BLK
        start = torch.cumsum(counts, 0) - counts
        pstart = torch.cumsum(padded, 0) - padded
        total = int(padded.sum())
        total_c = max(CHUNK, ((total + CHUNK - 1) // CHUNK) * CHUNK)
        dest = pstart[mol] + (torch.arange(len(mol)) - start[mol])
        for k, v in cols.items():
            fill = 1.0 if k == 'x2' else (0.5 if k == 'w' else 0.0)   # dummy rows: zero prefactor
            out = torch.full((total_c,), fill, dtype=torch.float32)
            out[dest] = v
            np.save(os.path.join(out_dir, f'{typ}_{k}.npy'), out.numpy())
        blk_mol = torch.zeros(total_c // BLK, dtype=torch.int64)
        blk_mol[: total // BLK] = torch.repeat_interleave(torch.arange(nmol), padded // BLK)
        np.save(os.path.join(out_dir, f'{typ}_blkmol.npy'), blk_mol.numpy())
        meta[f'{typ}_rows'] = int(len(mol))
        meta[f'{typ}_rows_padded'] = total_c
    sc = data.scalars
    base = (sc['E_tot'] - sc['E_sl_t64']).cpu().numpy()   # 'rsh' frame: HF terms identical to ωB97M-V
    np.save(os.path.join(out_dir, 'E_base.npy'), base)
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'))
    del data
    torch.cuda.empty_cache()
    log(f'syfes rows {split}: ' + ', '.join(f'{t} {meta[t + "_rows"] / 1e6:.1f}M' for t in ROW_TYPES)
        + f' ({time.time() - t0:.0f}s) -> {out_dir}')
    return out_dir


class Rows:
    """GPU-resident rows of one split + its reaction set."""

    def __init__(self, split, device, split_variant='syfes'):
        d = os.path.join(ROWS_DIR, split)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        if meta.get('layout') != 2:
            raise RuntimeError(f'old row layout in {d}; rebuild with build_rows')
        L = lambda f: torch.from_numpy(np.load(os.path.join(d, f))).to(device)
        self.split, self.device, self.nmol, self.names = split, device, meta['nmol'], meta['names']
        self.types = {}
        for t, pf in ROW_TYPES.items():
            arrs = {k: L(f'{t}_{k}.npy') for k in ('x2', 'w', *pf)}
            blk = L(f'{t}_blkmol.npy')
            nb = blk.numel()
            M = torch.sparse_coo_tensor(torch.stack([blk, torch.arange(nb, device=device)]),
                                        torch.ones(nb, dtype=torch.float64, device=device),
                                        (self.nmol, nb)).coalesce().to_sparse_csr()
            self.types[t] = (arrs, M)
        self.E_base = L('E_base.npy').double()
        self.n_rows = sum(meta[f'{t}_rows'] for t in ROW_TYPES)

        class _D:  # minimal SplitData stand-in for ReactionSet
            pass
        sd = _D()
        sd.index = {n: i for i, n in enumerate(self.names)}
        sd.nmol, sd.device = self.nmol, device
        R = load_reactions(split_variant)
        self.rs = ReactionSet([r for r in R if r.split == split], sd)


def ensure_rows(splits=('train', 'validation'), device='cuda', log=print):
    for sp in splits:
        if not os.path.exists(os.path.join(ROWS_DIR, sp, 'meta.json')):
            build_rows(sp, device, log=log)


# ============================================================================ kernels (codegen per form)
_TB = None


def _expr(ins, V, P, c):
    """Python expression for one instruction; V/P map local names to generated identifiers."""
    name, out, args = ins[0], ins[1], ins[2:]
    A = [V.get(a) or P.get(a) or a for a in args]   # features keep their names (x2, w)
    o = V[out]
    b = lambda n: P[n]
    tbl = {
        'AdditionBy1Instruction': f'{A[0]} + 1.0',
        'Power2Instruction': f'{A[0]} ** 2', 'Power3Instruction': f'{A[0]} ** 3',
        'Power4Instruction': f'{A[0]} ** 4', 'Power6Instruction': f'{A[0]} ** 6',
        'SquareRootInstruction': f'torch.sqrt({A[0]})', 'CubeRootInstruction': f'_cbrt({A[0]})',
        'Log1PInstruction': f'torch.log1p({A[0]})', 'ExpInstruction': f'torch.exp({A[0]})',
    }
    if name in tbl:
        return f'{o} = {tbl[name]}'
    if name == 'UTransformInstruction':
        return f'_t = {b("gamma_utransform")} * {A[0]}; {o} = _t / (1.0 + _t)'
    if name == 'PBEXInstruction':
        return f'{o} = S._f_x_pbe(_T, {A[0]}, {b("kappa_pbex")}, {b("mu_pbex")})'
    if name == 'RPBEXInstruction':
        return f'{o} = S._f_x_rpbe(_T, {A[0]}, {b("kappa_rpbex")}, {b("mu_rpbex")})'
    if name == 'B88XInstruction':
        return f'{o} = S._f_x_b88(_T, {A[0]}, {b("beta_b88x")})'
    if name == 'AdditionInstruction':
        return f'{o} = {A[0]} + {A[1]}'
    if name == 'SubtractionInstruction':
        return f'{o} = {A[0]} - {A[1]}'
    if name == 'MultiplicationInstruction':
        return f'{o} = {A[0]} * {A[1]}'
    if name == 'MultiplicationAdditionInstruction':
        return f'{o} = {o} + {A[0]} * {A[1]}'
    if name == 'DivisionInstruction':
        return f'{o} = {A[0]} / {A[1]}'
    if name == 'PBECInstruction':
        return f'{o} = S._e_c_pbe_unpolarized(_T, {A[0]}, {A[1]}, {b("beta_pbec")}, {b("gamma_pbec")})'
    raise ValueError(name)


def generate_source(form, layout, prec):
    """Source of kernels k_cl/k_oss/k_oos(theta, x2, w, *prefactors) -> block sums [P, n/BLK].
    prec: 'f32' (F, products and 128-row block sums in float32), 'f64' (F in float32, products and block sums in
    float64), 'full64' (everything float64; reference)."""
    pidx = {cn: k for k, cn in enumerate(layout)}

    def ef_lines(c, tag):
        ef = form[c]
        V = {v: f'{tag}_{v}' for v in ef['variable_names']}
        P = {n: f'theta[:, {pidx[(c, n)]}:{pidx[(c, n)] + 1}]' for n in S.ef_parameter_names(ef) if (c, n) in pidx}
        lines = [f'{V[v]} = _z' for v in ef['variable_names']]
        lines += [_expr(ins, V, P, c) for ins in ef['instructions']]
        return lines, V['enhancement_factor']

    up = '.double()' if prec == 'full64' else ''
    cv = (lambda e: f'({e}).double()') if prec in ('f64', 'full64') else (lambda e: e)
    comp = {'px': ('f_x', 'X'), 'pss': ('f_css', 'SS'), 'pos': ('f_cos', 'OS')}
    src = []
    for t, pf in ROW_TYPES.items():
        src += [f'def k_{t}(theta, x2, w, {", ".join(pf)}):', '    _z = theta.new_zeros(())',
                f'    x2 = x2[None, :]{up}', f'    w = w[None, :]{up}']
        terms = []
        for p_ in pf:
            lines, Fv = ef_lines(*comp[p_])
            src += ['    ' + ln for ln in lines]
            terms.append(f'torch.where({p_} == 0, 0.0, {cv(p_)} * {cv(Fv)})')
        src += ['    c = ' + ' + '.join(terms), f'    c = c.expand(theta.shape[0], {pf[0]}.shape[-1])']
        if prec == 'mix':   # float32 sums of SUB rows, then float64
            src += [f'    return c.reshape(theta.shape[0], -1, {BLK // SUB}, {SUB}).sum(-1).double().sum(-1)', '']
        else:
            src += [f'    return c.reshape(theta.shape[0], -1, {BLK}).sum(-1)', '']
    return '\n'.join(src)


def _cbrt(x):
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


class FormKernels:
    """Compiled kernels of one form, one set per precision mode (compiled lazily on first use)."""

    def __init__(self, form, layout, compile_=True):
        global _TB
        if _TB is None:
            _TB = S._torch_backend()
        self.form, self.layout, self.compile_ = form, layout, compile_
        self.sets = {}
        self.compiled = compile_
        self.compile_error = None

    def _get(self, prec):
        if prec not in self.sets:
            src = generate_source(self.form, self.layout, prec)
            g = {'torch': torch, 'S': S, '_T': _TB, '_cbrt': _cbrt}
            exec(compile(src, f'<syfes_form_{prec}_{abs(hash(src))}>', 'exec'), g)
            eager = {t: g[f'k_{t}'] for t in ROW_TYPES}
            comp = {t: torch.compile(f, dynamic=False) for t, f in eager.items()} if self.compiled else eager
            self.sets[prec] = dict(eager=eager, k=comp, src=src)
        return self.sets[prec]

    def __call__(self, prec, t, *args):
        ks = self._get(prec)
        try:
            return ks['k'][t](*args)
        except Exception as e:
            if not self.compiled:
                raise
            self.compile_error = f'{type(e).__name__}: {str(e)[:300]}'
            self.compiled = False
            for v in self.sets.values():
                v['k'] = v['eager']
            return ks['eager'][t](*args)


@torch.no_grad()
def energies(kern: FormKernels, theta: torch.Tensor, rows: Rows, prec='f64'):
    """Per-molecule total energies [nmol, P] (float64) for parameter batch theta [P, k]."""
    P = theta.shape[0]
    theta = theta.double() if prec == 'full64' else theta.float()
    E = rows.E_base[:, None].expand(rows.nmol, P).clone()
    for t, (arrs, M) in rows.types.items():
        keys = ('x2', 'w', *ROW_TYPES[t])
        n = arrs['x2'].numel()
        bs = torch.empty((P, n // BLK), dtype=torch.float32 if prec == 'f32' else torch.float64,
                         device=theta.device)
        for i0 in range(0, n, CHUNK):
            bs[:, i0 // BLK:(i0 + CHUNK) // BLK] = kern(prec, t, theta, *[arrs[k][i0:i0 + CHUNK] for k in keys])
        E += torch.sparse.mm(M, bs.double().T.contiguous())
    return E


@torch.no_grad()
def wrmsd(kern, theta, rows: Rows, prec='f64'):
    E = energies(kern, theta, rows, prec)                # [nmol, P]
    pred = torch.sparse.mm(rows.rs.S, E) * 627.509474    # [nr, P]
    err = pred - rows.rs.ref[:, None]
    return torch.sqrt(torch.mean(rows.rs.w[:, None] * err ** 2, dim=0))


# ============================================================================ CMA-ES training (SyFES optimizers.py)
CMA_DEFAULTS = dict(num_trials=5, sigma0=1.0, init_mean=0.0, init_std=1.0, bounds=(-10.0, 10.0), tolfun=1e-5,
                    maxfevals=100000, early_abnormal=1e5, early_wrmsd=50.0, early_fevals=5000)


def train_form(form, rows_train: Rows, rows_val: Rows | None = None, seed=0, opts=None, compile_=True, log=None):
    """CMA-ES on the training WRMSD (num_trials restarts from N(0,1), run in lockstep as one GPU batch); best trial's
    parameters -> validation WRMSD. Returns a JSON-serialisable dict."""
    import cma
    o = dict(CMA_DEFAULTS, **(opts or {}))
    dev = rows_train.device
    layout = S.parameter_layout(form)
    k = len(layout)
    t0 = time.time()
    res = dict(n_params=k, layout=[f'{c}.{n}' for c, n in layout])
    rng = np.random.RandomState(seed)
    if k == 0:
        lam, P = 1, 1
    else:
        lam = 4 + int(3 * math.log(k))
        P = lam * o['num_trials']
    kern = FormKernels(form, layout, compile_)
    nevals = 0

    prec_counts = {'f32': 0, 'f64': 0}

    def batch_loss(X, prec='f64'):
        nonlocal nevals
        n = len(X)
        th = np.zeros((P, max(k, 1)), dtype=np.float32)
        th[:n, :k] = X
        if n < P:
            th[n:, :k] = X[0]
        L = wrmsd(kern, torch.from_numpy(th).to(dev), rows_train, prec)[:n].cpu().numpy()
        nevals += n
        prec_counts[prec] += n
        return np.where(np.isfinite(L), L, BIG)

    if k == 0:
        best_x, best_f, trials = np.zeros(0), float(batch_loss(np.zeros((1, 0)))[0]), []
    else:
        ess = []
        for t in range(o['num_trials']):
            x0 = rng.normal(o['init_mean'], o['init_std'], size=k)
            x0 = np.clip(x0, o['bounds'][0] + 1e-9, o['bounds'][1] - 1e-9)
            ess.append(cma.CMAEvolutionStrategy(x0, o['sigma0'], {
                'bounds': list(o['bounds']), 'tolfun': o['tolfun'], 'maxfevals': o['maxfevals'],
                'seed': int(rng.randint(1, 2 ** 31 - 1)), 'verbose': -9, 'popsize': lam}))
        stop_reason = [None] * len(ess)
        last_range = [math.inf] * len(ess)
        while True:
            active = [i for i, es in enumerate(ess) if stop_reason[i] is None]
            if not active:
                break
            Xs = {i: ess[i].ask() for i in active}
            allX = np.concatenate([np.asarray(Xs[i]) for i in active])
            prec = 'f64' if min(last_range[i] for i in active) < SWITCH else 'f32'
            L = batch_loss(allX, prec)
            for j, i in enumerate(active):
                f = L[j * lam:(j + 1) * lam]
                last_range[i] = float(f.max() - f.min())
                es = ess[i]
                es.tell(Xs[i], f.tolist())
                if not np.any(f < o['early_abnormal']):
                    stop_reason[i] = 'abnormal_values'
                elif es.countevals >= o['early_fevals'] and es.best.f > o['early_wrmsd']:
                    stop_reason[i] = f'wrmsd>{o["early_wrmsd"]}@{o["early_fevals"]}'
                else:
                    st = es.stop()
                    if st:
                        stop_reason[i] = ','.join(st.keys())
        trials = [dict(f=float(es.best.f), evals=int(es.countevals), iters=int(es.countiter), stop=stop_reason[i])
                  for i, es in enumerate(ess)]
        ib = int(np.argmin([es.best.f for es in ess]))
        best_x, best_f = np.asarray(ess[ib].best.x, dtype=float), float(ess[ib].best.f)
    if k and best_f < BIG:   # report the best point's training loss with float64 accumulation
        th = np.zeros((P, k), dtype=np.float32)
        th[:] = best_x
        best_f = float(wrmsd(kern, torch.from_numpy(th).to(dev), rows_train, 'f64')[0])
    res.update(train_wrmsd=best_f, evals_by_precision=prec_counts, params=best_x.tolist(), trials=trials, n_evals=nevals, lam=lam, batch=P,
               train_seconds=round(time.time() - t0, 1), compiled=kern.compiled, compile_error=kern.compile_error)
    if rows_val is not None:
        t1 = time.time()
        if best_f >= BIG:
            res['validation_wrmsd'] = float('nan')
        else:
            th = np.zeros((P, max(k, 1)), dtype=np.float32)
            th[:, :k] = best_x
            v = float(wrmsd(kern, torch.from_numpy(th).to(dev), rows_val)[0])
            res['validation_wrmsd'] = v if math.isfinite(v) else float('nan')
        res['validation_seconds'] = round(time.time() - t1, 1)
    return res


def evaluate_params(form, theta, rows: Rows, compile_=False):
    layout = S.parameter_layout(form)
    kern = FormKernels(form, layout, compile_)
    th = torch.as_tensor(np.asarray(theta, dtype=np.float32).reshape(1, -1), device=rows.device)
    if th.shape[1] == 0:
        th = torch.zeros((1, 1), device=rows.device)
    return float(wrmsd(kern, th, rows)[0])


# ============================================================================ Functional wrapper (fp64 checks)
def as_functional(form):
    """dftsr.api.Functional evaluating a SyFES form (params p = {'theta': tensor[k]}), for the float64 evaluator and
    the constraint suite."""
    from .api import Functional, exchange_from_fx, correlation_from_factors
    layout = S.parameter_layout(form)
    TB = S._torch_backend()

    def P(p, c):
        th = p['theta']
        return {n: th[k] for k, (cc, n) in enumerate(layout) if cc == c}

    class SyFESForm(Functional):
        family = 'rsh'

        def init_params(self):
            return dict(theta=[0.0] * len(layout))

        def exchange(self, p, d):
            z = p['theta'].new_zeros(()) if len(layout) else d.rho_a.new_zeros(())
            return exchange_from_fx(d, lambda ch: S.eval_ef(form['f_x'], {'x2': ch.x2, 'w': ch.w}, P(p, 'f_x'), TB, z)
                                    + 0 * ch.rho, self.omega)

        def correlation(self, p, d):
            z = p['theta'].new_zeros(()) if len(layout) else d.rho_a.new_zeros(())
            fss = lambda ch: S.eval_ef(form['f_css'], {'x2': ch.x2, 'w': ch.w}, P(p, 'f_css'), TB, z) + 0 * ch.rho
            fos = lambda dd: S.eval_ef(form['f_cos'], {'x2': dd.x2_avg, 'w': dd.w_avg}, P(p, 'f_cos'), TB, z) \
                + 0 * dd.rho_a
            return correlation_from_factors(d, fss, fos)
    return SyFESForm()
