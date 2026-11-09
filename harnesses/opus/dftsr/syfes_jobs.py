"""SyFES baseline jobs: registry (runs/syfes/jobs.json), history summaries and reports (no GPU needed)."""
from __future__ import annotations

import json
import math
import os
import time

from .config import RUNS_DIR

SYFES_DIR = os.path.join(RUNS_DIR, 'syfes')
REG = os.path.join(SYFES_DIR, 'jobs.json')


def _load():
    if not os.path.exists(REG):
        return []
    return json.load(open(REG))


def _save(jobs):
    os.makedirs(SYFES_DIR, exist_ok=True)
    tmp = REG + '.tmp'
    json.dump(jobs, open(tmp, 'w'), indent=1)
    os.replace(tmp, REG)


def create_job(**kw):
    jobs = _load()
    jid = 1 + max([j['id'] for j in jobs], default=0)
    jobs.append(dict(id=jid, created=time.strftime('%Y-%m-%d %H:%M:%S'), status='created', **kw))
    _save(jobs)
    return jid


def update_job(jid, **kw):
    jobs = _load()
    for j in jobs:
        if j['id'] == jid:
            j.update(kw)
    _save(jobs)


def get_job(jid):
    return next((j for j in _load() if j['id'] == jid), None)


def list_jobs():
    return _load()


def read_history(outdir):
    p = os.path.join(outdir, 'history.jsonl')
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def summarize(outdir, top=10, with_forms=False, symbolic=False):
    from . import syfes as S
    H = read_history(outdir)
    seed = next((r for r in H if r['ev'] == 'seed'), None)
    trained = [r for r in H if r['ev'] == 'trained']
    ok = [r for r in trained if r.get('status') == 'ok' and _fin(r.get('validation_wrmsd'))]
    best, traj = math.inf, []
    for i, r in enumerate(trained, 1):
        v = r.get('validation_wrmsd')
        if _fin(v) and v < best:
            best = v
            traj.append(dict(n_trained=i, form=r['form'], validation=round(v, 4), train=round(r['train_wrmsd'], 4)))
    seen, ranked = set(), []
    for r in sorted(ok, key=lambda r: r['validation_wrmsd']):
        if r['fingerprint'] in seen:
            continue
        seen.add(r['fingerprint'])
        ranked.append(r)
    cnt = {}
    for r in H:
        cnt[r['ev']] = cnt.get(r['ev'], 0) + 1
    secs = [r['seconds'] for r in trained if r.get('seconds')]
    evals = [r['n_evals'] for r in trained if r.get('n_evals')]
    out = dict(
        seed=None if not seed else dict(train=seed['train_wrmsd'], validation=seed['validation_wrmsd']),
        n_trained=len(trained), n_finite=len(ok), n_nonfinite_or_error=len(trained) - len(ok),
        n_equivalent_mutations=cnt.get('equivalent', 0), n_inflight_duplicates=cnt.get('dup_inflight', 0),
        n_readds=cnt.get('readd', 0),
        n_better_than_seed_val=sum(1 for r in ok if seed and r['validation_wrmsd'] < seed['validation_wrmsd']),
        best_trajectory=traj,
        gpu_seconds_total=round(sum(secs)), mean_seconds_per_form=round(sum(secs) / len(secs), 1) if secs else None,
        mean_cma_evals_per_form=round(sum(evals) / len(evals)) if evals else None,
        first_event=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(H[0]['t'])) if H else None,
        last_event=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(H[-1]['t'])) if H else None,
    )
    forms = {r['form']: r['form_def'] for r in H if 'form_def' in r}
    tops = []
    for r in ranked[:top]:
        d = dict(form=r['form'], parent=r['parent'], validation=round(r['validation_wrmsd'], 4),
                 train=round(r['train_wrmsd'], 4), n_params=r['n_params'], mutation=f"{r['component']} {r['change']}")
        if with_forms:
            d['instructions'] = S.form_to_text(forms[r['form']])
            d['params'] = dict(zip(r['layout'], [round(x, 6) for x in r['params']]))
        if symbolic:
            d['expression'] = S.symbolic_expression(forms[r['form']])
        tops.append(d)
    out['top'] = tops
    rc = os.path.join(outdir, 'recheck.json')
    if os.path.exists(rc):
        out['recheck_fp64'] = json.load(open(rc))
    return out


def lineage(outdir, form_id):
    H = read_history(outdir)
    par = {r['form']: r for r in H if r['ev'] in ('dispatch', 'seed')}
    # forms created by a fingerprint-equivalent mutation are logged as 'register' + 'equivalent' (member=form)
    for r in H:
        if r['ev'] == 'equivalent' and r.get('member') is not None and r['member'] not in par:
            par[r['member']] = dict(r, form=r['member'], validation_wrmsd=r.get('fitness'),
                                    change=f"{r.get('change', '')} (equivalent to an existing form)")
    tr = {r['form']: r for r in H if r['ev'] == 'trained'}
    chain, f = [], form_id
    while f is not None and f in par:
        r = par[f]
        t = tr.get(f, {})
        chain.append(dict(form=f, parent=r.get('parent'), change=f"{r.get('component', '')} {r.get('change', 'seed')}",
                          validation=t.get('validation_wrmsd', r.get('validation_wrmsd'))))
        f = r.get('parent')
        if len(chain) > 500:
            break
    return chain
