#!/usr/bin/env python
"""Establish the reference numbers and seed iterations once train+validation densities/caches exist.

  * ωB97M-V with published parameters (no refit)  -> baseline_val_wrmsd, J_target = 0.858 x baseline
    (Duston et al.: J_target = 3.45 vs ωB97M-V validation 4.02 kcal/mol)
  * seed iterations (refit on train): ωB97M-V, GAS22, B97M-V  -> iterations table (island 'seed')
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dftsr import tracker
from dftsr.worker import Workspace, evaluate
from dftsr.config import PROJECT_ROOT

TARGET_RATIO = 3.45 / 4.02

ws = Workspace('cuda:0')
con = tracker.connect()
base = evaluate(os.path.join(PROJECT_ROOT, 'baselines', 'wb97mv.py'), ws, fit=False)
assert base['status'] == 'ok', base
tracker.set_meta(con, 'baseline_val_wrmsd', str(base['val_wrmsd']))
tracker.set_meta(con, 'baseline_train_wrmsd', str(base['train_wrmsd']))
tracker.set_meta(con, 'j_target', str(TARGET_RATIO * base['val_wrmsd']))
tracker.set_meta(con, 'baseline_published_params', json.dumps(dict(
    train=base['train_wrmsd'], val=base['val_wrmsd'], val_by_type=base['validation']['by_type'],
    constraints=base['constraints'])))
print(f"ωB97M-V (published params): train {base['train_wrmsd']:.3f}  val {base['val_wrmsd']:.3f} kcal/mol")
j_target = TARGET_RATIO * base['val_wrmsd']

for fn, name in (('wb97mv.py', 'wB97M-V (refit)'), ('gas22.py', 'GAS22 (refit)'), ('b97mv.py', 'B97M-V (refit)')):
    path = os.path.join(PROJECT_ROOT, 'baselines', fn)
    if con.execute('SELECT 1 FROM iterations WHERE name=? AND island="seed"', (name,)).fetchone():
        continue
    r = evaluate(path, ws, fit=True, n_restarts=2, j_target=j_target)
    con.execute('INSERT INTO iterations (created, name, parent_id, island, family, status, plan, code, code_path, '
                'n_params, params_json, train_wrmsd, val_wrmsd, score, n_violations, constraints_json, metrics_json, '
                'eval_seconds, tokens_total, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (tracker.now(), name, None, 'seed', r.get('family'), r['status'],
                 f'Seed: {name} functional form from the literature, parameters refit on MGCDB84 train.',
                 open(path).read(), path, r.get('n_params'), json.dumps(r.get('params')), r.get('train_wrmsd'),
                 r.get('val_wrmsd'), r.get('score'), r.get('n_violations'), json.dumps(r.get('constraints')),
                 json.dumps({k: r.get(k) for k in ('train', 'validation', 'fit')}), r.get('eval_seconds'), 0, 'seed'))
    con.commit()
    print(f"{name}: train {r.get('train_wrmsd')}  val {r.get('val_wrmsd')}  score {r.get('score')}  "
          f"violations {r.get('n_violations')}  ({r.get('eval_seconds')} s)")
