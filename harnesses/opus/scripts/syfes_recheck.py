#!/usr/bin/env python
"""Float64 re-check of the best SyFES forms of a run: exact train/validation WRMSD with the standard non-SCF
evaluator (dftsr/evaluator.py, no refit, stored CMA-ES parameters) + the 4 physical constraint tests used for the
LLM campaigns. Writes <out>/recheck.json and <out>/REPORT.md. Test-split metrics are NOT computed here
(the test split stays gated, as for the LLM campaigns)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', type=int, default=10)
    a = ap.parse_args()
    import torch
    from dftsr import syfes as S, syfes_gpu as G, syfes_jobs as J
    from dftsr.worker import Workspace
    from dftsr.evaluator import Evaluator
    from dftsr.constraints import ConstraintSuite
    H = J.read_history(a.out)
    forms = {r['form']: r['form_def'] for r in H if 'form_def' in r}
    seed = next(r for r in H if r['ev'] == 'seed')
    ok = [r for r in H if r['ev'] == 'trained' and r.get('status') == 'ok' and r.get('validation_wrmsd') is not None
          and math.isfinite(r['validation_wrmsd'])]
    seen, cands = set(), []
    for r in sorted(ok, key=lambda r: r['validation_wrmsd']):
        if r['fingerprint'] not in seen:
            seen.add(r['fingerprint'])
            cands.append(r)
    cands = [dict(seed, form=0, n_params=len(seed['params']))] + cands[:a.top]
    dev = 'cuda'
    ws = Workspace(dev, splits=('train', 'validation'))
    cs = ConstraintSuite(dev)
    out = []
    for r in cands:
        form = forms[r['form']]
        F = G.as_functional(form)
        p = {'theta': torch.tensor(r['params'], dtype=torch.float64, device=dev)}
        ev = Evaluator(F, ws.data, compile_=False)
        res = dict(form=r['form'], n_params=len(r['params']),
                   search_train=r['train_wrmsd'], search_validation=r['validation_wrmsd'])
        for sp in ('train', 'validation'):
            E = ev.total_energies(p, ws.data[sp], torch.float64)
            res[f'{sp}_fp64'] = float(ws.rs[sp].wrmsd(E))
        c = cs.run(F, p)
        res['n_violations'] = c['n_violations']
        res['constraints'] = {k: dict(value=v.get('value'), passed=v.get('passed')) for k, v in c.items()
                              if isinstance(v, dict)}
        res['expression'] = S.symbolic_expression(form)
        res['instructions'] = S.form_to_text(form)
        res['params'] = r['params']
        res['layout'] = [f'{c_}.{n}' for c_, n in S.parameter_layout(form)]
        out.append(res)
        print(f"form {r['form']}: train {res['train_fp64']:.4f} val {res['validation_fp64']:.4f} "
              f"violations {res['n_violations']}", flush=True)
    json.dump(out, open(os.path.join(a.out, 'recheck.json'), 'w'), indent=1)
    s = J.summarize(a.out, top=a.top)
    L = ['# SyFES baseline run (non-LLM regularized evolution from GAS22)', '',
         f'Run directory: `{a.out}`', '',
         f"Trained forms: {s['n_trained']} (finite {s['n_finite']}, non-finite/error {s['n_nonfinite_or_error']}); "
         f"equivalent mutations skipped by fingerprint: {s['n_equivalent_mutations']}; "
         f"GPU time {s['gpu_seconds_total'] / 3600:.1f} GPU-h, mean {s['mean_seconds_per_form']} s and "
         f"{s['mean_cma_evals_per_form']} CMA-ES evaluations per form.", '',
         '## Best forms, float64 re-check (stored parameters, no refit; SyFES split)', '',
         '| form | params | train | validation | violations | failed constraints |', '|---|---|---|---|---|---|']
    for r in out:
        failed = ', '.join(k for k, v in r['constraints'].items() if not v['passed'])
        L.append(f"| {'GAS22 (seed, published)' if r['form'] == 0 else r['form']} | {r['n_params']} | "
                 f"{r['train_fp64']:.4f} | {r['validation_fp64']:.4f} | {r['n_violations']} | {failed} |")
    L += ['', '## Best-validation trajectory (search numbers)', '', '| trained forms | form | train | validation |',
          '|---|---|---|---|']
    L += [f"| {t['n_trained']} | {t['form']} | {t['train']} | {t['validation']} |" for t in s['best_trajectory']]
    L += ['', '## Expressions of the re-checked forms', '']
    for r in out:
        L += [f"### form {r['form']}", '', '```', r['instructions'], '```', '',
              *[f"- {k}: `{v}`" for k, v in r['expression'].items()],
              '- params: ' + ', '.join(f'{n}={x:.6g}' for n, x in zip(r['layout'], r['params'])), '']
    open(os.path.join(a.out, 'REPORT.md'), 'w').write('\n'.join(L))
    print('wrote', os.path.join(a.out, 'REPORT.md'))


if __name__ == '__main__':
    main()
