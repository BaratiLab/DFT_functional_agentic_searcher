#!/usr/bin/env python
"""Progress report for the ωB97M-V density campaign (per split, failures, ETA)."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dftsr.config import DENSITY_DIR
from dftsr.dataset import load_reactions, molecules_for

done = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(DENSITY_DIR, '*.h5'))}
logs = []
for p in glob.glob(os.path.join(DENSITY_DIR, '_logs', '*.jsonl')):
    logs += [json.loads(l) for l in open(p)]
errors = {}
for r in logs:
    if r.get('status') == 'error' and r['name'] not in done:
        errors[r['name']] = r['error']
unconv = [r['name'] for r in logs if r.get('status') == 'ok' and not r.get('converged')]
running = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DENSITY_DIR, '_locks', '*.lock'))]

R = load_reactions()
seen = set()
for sp in ('train', 'validation', 'test'):
    ms = [m for m in molecules_for([r for r in R if r.split == sp])]
    n_done = sum(m in done for m in ms)
    print(f'{sp:11s} molecules {n_done:5d}/{len(ms):5d} ({100 * n_done / len(ms):5.1f}%)')
ok = [r for r in logs if r.get('status') == 'ok']
if ok:
    t_gpu = sum(r['t_total'] for r in ok)
    print(f'completed {len(ok)} molecules, {t_gpu / 3600:.2f} GPU-hours so far')
    if len(ok) > 1:
        ts = sorted(time.mktime(time.strptime(r['time'], '%Y-%m-%d %H:%M:%S')) for r in ok)
        print(f'wall time since first completion: {(ts[-1] - ts[0]) / 3600:.2f} h')
print('running now:', running)
print('unconverged (saved, flagged):', len(unconv), unconv[:10])
print('errors (not yet done):', len(errors))
for k, v in list(errors.items())[:10]:
    print('   ', k, v[:200])
