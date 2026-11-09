#!/usr/bin/env python
"""Non-LLM baseline: SyFES regularized evolution (Ma et al. 2022) on the MGCDB84 non-SCF caches.

Population server (this process) + one GPU worker process per GPU (asynchronous, like the paper's 50 workers):
  parent  = tournament winner (tournament 25 of population <= 100; with probability 0.1 the winner is first re-added
            to the population, regularized_evolution.Population.get_parent)
  child   = one SyFES mutation of one randomly chosen enhancement factor
  fingerprint hit (functional equivalence) -> child re-enters the population with the cached fitness, no training
  otherwise -> CMA-ES fit on train (5 restarts from N(0,1), bounds [-10,10]) on a GPU worker; fitness = validation
            WRMSD of the best-train restart; non-finite children are dropped; the oldest individual is removed once
            the population exceeds its size.
The run stops after --n-forms TRAINED forms. Everything is appended to <out>/history.jsonl (resumable: rerunning
the same command replays it); <out>/progress.json is rewritten after every event.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def worker_main(gpu, inq, outq):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    from dftsr import syfes_gpu as G
    try:
        t0 = time.time()
        tr = G.Rows('train', 'cuda')
        va = G.Rows('validation', 'cuda')
        outq.put(dict(kind='ready', gpu=gpu, seconds=round(time.time() - t0, 1)))
    except Exception as e:
        outq.put(dict(kind='fatal', gpu=gpu, error=repr(e), traceback=traceback.format_exc()))
        return
    while True:
        job = inq.get()
        if job is None:
            return
        t0 = time.time()
        try:
            if job['kind'] == 'train':
                r = G.train_form(job['form'], tr, va, seed=job['seed'], opts=job.get('opts'))
            else:   # evaluate fixed parameters
                r = dict(train_wrmsd=G.evaluate_params(job['form'], job['theta'], tr),
                         validation_wrmsd=G.evaluate_params(job['form'], job['theta'], va))
            r.update(status='ok')
        except Exception as e:
            r = dict(status='error', error=f'{type(e).__name__}: {str(e)[:500]}', traceback=traceback.format_exc()[-2000:])
            torch.cuda.empty_cache()
        r.update(kind='result', id=job['id'], gpu=gpu, seconds=round(time.time() - t0, 1))
        outq.put(r)


class Run:
    def __init__(self, a):
        from dftsr import syfes as S
        self.S, self.a = S, a
        os.makedirs(a.out, exist_ok=True)
        self.hist_path = os.path.join(a.out, 'history.jsonl')
        self.pop = collections.OrderedDict()      # individual uid -> (fitness, form id)
        self.forms = {}                           # form id -> form dict
        self.fps = {}                             # fingerprint -> fitness (nan if non-finite)
        self.inflight = {}                        # form id -> job
        self.counters = collections.Counter()
        self.best = (math.inf, None)
        self.next_id = 0
        self.uid = 0
        self.t_start = time.time()
        self.events = 0
        self.replayed = False

    # ---------------------------------------------------------------- bookkeeping
    def log(self, rec):
        rec['t'] = round(time.time(), 2)
        with open(self.hist_path, 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n')
        self.events += 1

    def add_to_population(self, fitness, fid):
        self.uid += 1
        self.pop[self.uid] = (fitness, fid)
        if len(self.pop) > self.a.population:
            self.pop.popitem(last=False)          # remove the oldest individual

    def tournament(self):
        items = list(self.pop.values())
        k = min(len(items), self.a.tournament)
        return min(self.rng.sample(items, k), key=lambda x: x[0])

    def get_parent(self):
        while self.rng.uniform(0, 1) > self.a.mutation_prob:
            fit, fid = self.tournament()
            self.add_to_population(fit, fid)
            self.log(dict(ev='readd', form=fid, fitness=fit))
            self.counters['readd'] += 1
        return self.tournament()[1]

    def progress(self, final=False):
        n_done = self.counters['trained']
        el = time.time() - self.t_start
        rate = (el / self.counters['trained_this_session']) if self.counters['trained_this_session'] else None
        remaining = self.a.n_forms - n_done
        eta = rate * remaining if rate else None     # rate = wall seconds per trained form (all GPUs together)
        p = dict(n_forms_target=self.a.n_forms, n_trained=n_done, n_inflight=len(self.inflight),
                 n_equivalent_skipped=self.counters['equivalent'], n_inflight_duplicates=self.counters['dup_inflight'],
                 n_nonfinite=self.counters['nonfinite'], n_errors=self.counters['error'],
                 n_readds=self.counters['readd'], population=len(self.pop),
                 best_validation=self.best[0] if math.isfinite(self.best[0]) else None, best_form=self.best[1],
                 seed_validation=self.seed_val, elapsed_s=round(el), eta_s=round(eta) if eta else None,
                 finished=final, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(p, open(os.path.join(self.a.out, 'progress.json.tmp'), 'w'), indent=1)
        os.replace(os.path.join(self.a.out, 'progress.json.tmp'), os.path.join(self.a.out, 'progress.json'))

    # ---------------------------------------------------------------- replay (resume)
    def replay(self):
        if not os.path.exists(self.hist_path):
            return False
        dispatched = {}
        for line in open(self.hist_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = r['ev']
            self.events += 1
            if ev == 'seed':
                self.forms[r['form']] = r['form_def']
                self.seed_val = r['validation_wrmsd']
                self.add_to_population(r['validation_wrmsd'], r['form'])
                self.fps[r['fingerprint']] = r['validation_wrmsd']
                self.best = (r['validation_wrmsd'], r['form'])
            elif ev == 'dispatch':
                self.forms[r['form']] = r['form_def']
                dispatched[r['form']] = r
            elif ev == 'trained':
                dispatched.pop(r['form'], None)
                self.counters['trained'] += 1
                v = r.get('validation_wrmsd')
                ok = r.get('status') == 'ok' and v is not None and math.isfinite(v)
                self.fps[r['fingerprint']] = v if ok else float('nan')
                if ok:
                    self.add_to_population(v, r['form'])
                    if v < self.best[0]:
                        self.best = (v, r['form'])
                else:
                    self.counters['nonfinite' if r.get('status') == 'ok' else 'error'] += 1
            elif ev == 'register':
                self.forms[r['form']] = r['form_def']
            elif ev == 'equivalent':
                self.counters['equivalent'] += 1
                if r.get('member') is not None:
                    self.add_to_population(r['fitness'], r['member'])
            elif ev == 'readd':
                self.counters['readd'] += 1
                self.add_to_population(r['fitness'], r['form'])
            elif ev == 'dup_inflight':
                self.counters['dup_inflight'] += 1
            self.next_id = max(self.next_id, r.get('form', -1) + 1 if isinstance(r.get('form'), int) else 0)
        self.redo = list(dispatched.values())
        return True

    # ---------------------------------------------------------------- main loop
    def run(self):
        a, S = self.a, self.S
        self.rng = random.Random(a.seed)
        self.seed_val = None
        resumed = self.replay()
        self.rng = random.Random(a.seed * 1000003 + self.events)
        self.mut = S.Mutator(seed=(a.seed * 7919 + self.events) % (2 ** 31))
        json.dump(vars(a), open(os.path.join(a.out, 'run_config.json'), 'w'), indent=1)
        gpus = [int(g) for g in a.gpus.split(',')]
        ctx = mp.get_context('spawn')
        outq = ctx.Queue()
        inqs = {g: ctx.Queue() for g in gpus}
        procs = {g: ctx.Process(target=worker_main, args=(g, inqs[g], outq), daemon=True) for g in gpus}
        for p in procs.values():
            p.start()
        ready = set()
        while len(ready) < len(gpus):
            m = outq.get()
            if m['kind'] == 'fatal':
                raise RuntimeError(f'worker {m["gpu"]} failed: {m["error"]}\n{m["traceback"]}')
            ready.add(m['gpu'])
            print(f'worker gpu{m["gpu"]} ready in {m["seconds"]}s', flush=True)
        free = list(gpus)
        busy = {}

        if not resumed:
            seed_form = S.GAS22_FORM
            theta = S.params_to_theta(seed_form, S.GAS22_PARAMS).tolist()
            inqs[free[0]].put(dict(kind='eval', id=-1, form=seed_form, theta=theta))
            r = outq.get()
            if r['status'] != 'ok':
                raise RuntimeError(f'seed evaluation failed: {r}')
            self.forms[0] = seed_form
            self.seed_val = r['validation_wrmsd']
            fp = S.fingerprint(seed_form)
            self.fps[fp] = r['validation_wrmsd']
            self.add_to_population(r['validation_wrmsd'], 0)
            self.best = (r['validation_wrmsd'], 0)
            self.log(dict(ev='seed', form=0, name='GAS22 (published parameters)', form_def=seed_form,
                          fingerprint=fp, params=theta, train_wrmsd=r['train_wrmsd'],
                          validation_wrmsd=r['validation_wrmsd']))
            self.next_id = 1
            self.redo = []
            print(f'seed GAS22: train {r["train_wrmsd"]:.4f} val {r["validation_wrmsd"]:.4f}', flush=True)
        self.progress()

        def dispatch(g, fid, form, meta):
            job = dict(kind='train', id=fid, form=form, seed=a.seed * 100003 + fid,
                       opts=dict(num_trials=a.trials, maxfevals=a.maxfevals))
            self.inflight[fid] = dict(job=job, meta=meta, t=time.time())
            busy[g] = fid
            inqs[g].put(job)

        n_dispatched = self.counters['trained'] + len(self.redo)
        for r in self.redo:                       # resume: forms dispatched but never finished
            if not free:
                break
            g = free.pop(0)
            meta = {k: r[k] for k in ('parent', 'component', 'mutation', 'change', 'fingerprint')}
            dispatch(g, r['form'], r['form_def'], meta)
        self.redo = [r for r in self.redo if r['form'] not in self.inflight]
        inflight_fps = {v['meta']['fingerprint'] for v in self.inflight.values()}

        while True:
            # fill free workers
            while free and n_dispatched < a.n_forms:
                if self.redo:
                    r = self.redo.pop(0)
                    g = free.pop(0)
                    dispatch(g, r['form'], r['form_def'],
                             {k: r[k] for k in ('parent', 'component', 'mutation', 'change', 'fingerprint')})
                    inflight_fps.add(r['fingerprint'])
                    n_dispatched += 1
                    continue
                tries = 0
                while True:
                    tries += 1
                    if tries > 100000:
                        raise RuntimeError('100000 consecutive equivalent mutations')
                    pid = self.get_parent()
                    child, info = self.mut.mutate(self.forms[pid])
                    fp = S.fingerprint(child)
                    if fp in self.fps:
                        fit = self.fps[fp]
                        self.counters['equivalent'] += 1
                        member = None
                        if fit is not None and math.isfinite(fit):
                            member = (pid if S.form_key(child) == S.form_key(self.forms[pid])
                                      else self._register(child))
                            self.add_to_population(fit, member)
                        self.log(dict(ev='equivalent', parent=pid, member=member, fingerprint=fp, fitness=fit,
                                      **info))
                        continue
                    if fp in inflight_fps:
                        self.counters['dup_inflight'] += 1
                        self.log(dict(ev='dup_inflight', parent=pid, fingerprint=fp, **info))
                        continue
                    break
                fid = self.next_id
                self.next_id += 1
                self.forms[fid] = child
                meta = dict(parent=pid, fingerprint=fp, **info)
                self.log(dict(ev='dispatch', form=fid, form_def=child, n_params=len(S.parameter_layout(child)),
                              **meta))
                g = free.pop(0)
                dispatch(g, fid, child, meta)
                inflight_fps.add(fp)
                n_dispatched += 1
            self.progress()
            if not self.inflight:
                break
            r = outq.get()
            if r.get('kind') == 'fatal':
                raise RuntimeError(f'worker failed: {r}')
            fid = r['id']
            job = self.inflight.pop(fid)
            g = r['gpu']
            busy.pop(g, None)
            free.append(g)
            meta = job['meta']
            inflight_fps.discard(meta['fingerprint'])
            v = r.get('validation_wrmsd')
            ok = r['status'] == 'ok' and v is not None and math.isfinite(v)
            self.counters['trained'] += 1
            self.counters['trained_this_session'] += 1
            self.fps[meta['fingerprint']] = v if ok else float('nan')
            if ok:
                self.add_to_population(v, fid)
                if v < self.best[0]:
                    self.best = (v, fid)
            else:
                self.counters['nonfinite' if r['status'] == 'ok' else 'error'] += 1
            rec = dict(ev='trained', form=fid, **meta, gpu=g, status=r['status'],
                       train_wrmsd=r.get('train_wrmsd'), validation_wrmsd=v, n_params=r.get('n_params'),
                       params=r.get('params'), layout=r.get('layout'), trials=r.get('trials'),
                       n_evals=r.get('n_evals'), seconds=r.get('seconds'), compiled=r.get('compiled'),
                       error=r.get('error'))
            self.log(rec)
            tv = f'{v:.4f}' if ok else 'nan'
            tt = r.get('train_wrmsd')
            print(f'[{self.counters["trained"]}/{a.n_forms}] form {fid} (parent {meta["parent"]}, '
                  f'{meta["component"]} {meta["mutation"]}) gpu{g} {r.get("seconds")}s evals {r.get("n_evals")} '
                  f'train {tt if tt is None else round(tt, 4)} val {tv} | best {self.best[0]:.4f} (form '
                  f'{self.best[1]})', flush=True)
        for q in inqs.values():
            q.put(None)
        self.progress(final=True)
        print('finished', flush=True)

    def _register(self, form):
        """An equivalent-but-different child enters the population under its own id (so it can be mutated)."""
        fid = self.next_id
        self.next_id += 1
        self.forms[fid] = form
        self.log(dict(ev='register', form=fid, form_def=form))
        return fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-forms', type=int, default=400)
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--population', type=int, default=100)
    ap.add_argument('--tournament', type=int, default=25)
    ap.add_argument('--mutation-prob', type=float, default=0.9)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--maxfevals', type=int, default=100000, help='CMA-ES max evaluations per restart (SyFES 1e5)')
    a = ap.parse_args()
    Run(a).run()


if __name__ == '__main__':
    main()
