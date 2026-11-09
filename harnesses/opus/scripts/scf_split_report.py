"""Per-split report of an SCF job directory (reads <outdir>/molecules.jsonl; does not touch the GPU).

For each split present: SCF and non-SCF WRMSD of the functional (overall + per data type), the ωB97M-V
reference (its stored self-consistent E_tot) on the same reactions, val+test without RG10, density-driven
error statistics |E_SCF - E_nonSCF| per molecule, and the names of failed / unconverged molecules.
Only reactions whose molecules all finished with status ok are scored (counts are reported).

Usage: python scripts/scf_split_report.py <outdir> [--subset probe|all|...] [--json out.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dftsr.dataset import HARTREE_TO_KCAL  # noqa: E402


def stats(reacts, E):
    rows = [(r, sum(c * E[m] for c, m in r.stoich) * HARTREE_TO_KCAL) for r in reacts
            if all(m in E for _, m in r.stoich)]
    if not rows:
        return None
    by = {}
    for r, v in rows:
        by.setdefault(r.dtype, []).append((r.weight, v - r.ref * HARTREE_TO_KCAL))
    w = np.array([r.weight for r, _ in rows])
    e = np.array([v - r.ref * HARTREE_TO_KCAL for r, v in rows])
    return dict(n=len(rows), wrmsd=float(np.sqrt(np.mean(w * e ** 2))),
                by_type={k: dict(n=len(v), wrmsd=float(np.sqrt(np.mean([a * b ** 2 for a, b in v]))))
                         for k, v in sorted(by.items())})


def strip_rg(m):
    bt = {k: v for k, v in m['by_type'].items() if k != 'RG10'}
    n = sum(v['n'] for v in bt.values())
    return dict(n=n, wrmsd=math.sqrt(sum(v['n'] * v['wrmsd'] ** 2 for v in bt.values()) / n)) if n else None


def combined(ms):
    ms = [m for m in ms if m]
    n = sum(m['n'] for m in ms)
    return math.sqrt(sum(m['n'] * m['wrmsd'] ** 2 for m in ms) / n) if n else None


def report(outdir, subset):
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from scf_evaluate import select
    import h5py
    from dftsr.features import density_path
    reacts, mols = select(subset)
    recs = {}
    path = os.path.join(outdir, 'molecules.jsonl')
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if recs.get(r['name'], {}).get('status') != 'ok':   # keep an ok record over an earlier error
                recs[r['name']] = r
    ok = {n: r for n, r in recs.items() if r.get('status') == 'ok'}
    nonfinite = [n for n, r in ok.items() if not all(math.isfinite(r.get(k, float('nan')))
                                                      for k in ('E_scf', 'E_nonscf'))]
    ok = {n: r for n, r in ok.items() if n not in nonfinite}
    E_scf = {n: r['E_scf'] for n, r in ok.items()}
    E_non = {n: r['E_nonscf'] for n, r in ok.items()}
    E_ref = {}
    for n in ok:
        with h5py.File(density_path(n), 'r') as f:
            E_ref[n] = float(f.attrs['E_tot'])
    dd = np.array([abs(E_scf[n] - E_non[n]) * HARTREE_TO_KCAL for n in ok])
    top = sorted(ok, key=lambda n: -abs(E_scf[n] - E_non[n]))[:5]
    out = dict(outdir=outdir, subset=subset, n_molecules=len(mols), n_records=len(recs), n_ok=len(ok),
               errors=sorted(n for n, r in recs.items() if r.get('status') == 'error'),
               nonfinite=sorted(nonfinite),
               unconverged=sorted(n for n, r in ok.items() if not r.get('converged')),
               missing=len(set(mols) - set(recs)),
               density_driven_kcal=dict(mean=float(dd.mean()) if len(dd) else None,
                                        median=float(np.median(dd)) if len(dd) else None,
                                        max=float(dd.max()) if len(dd) else None,
                                        top5={n: round(abs(E_scf[n] - E_non[n]) * HARTREE_TO_KCAL, 3) for n in top}),
               mol_seconds=dict(total=round(sum(r.get('t', 0) for r in recs.values()), 1),
                                max=max((r.get('t', 0) for r in recs.values()), default=None)),
               splits={})
    for sp in ('train', 'validation', 'test'):
        rs = [r for r in reacts if r.split == sp]
        if not rs:
            continue
        out['splits'][sp] = dict(n_reactions=len(rs), scf=stats(rs, E_scf), nonscf=stats(rs, E_non),
                                 wb97mv=stats(rs, E_ref))
    s = out['splits']
    if 'validation' in s and 'test' in s:
        out['valtest_noRG10'] = {k: combined([strip_rg(s[sp][k]) for sp in ('validation', 'test') if s[sp][k]])
                                 for k in ('scf', 'nonscf', 'wb97mv')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--subset', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    subset = a.subset
    if subset is None:
        subset = json.load(open(os.path.join(a.outdir, 'progress.json')))['subset']
    out = report(a.outdir, subset)
    txt = json.dumps(out, indent=1)
    if a.json:
        open(a.json, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
