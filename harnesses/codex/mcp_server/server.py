#!/usr/bin/env python
"""MCP server: agentic discovery of XC functionals on MGCDB84 (non-SCF screening + gated SCF validation).

Tools (non-SCF):   get_task_brief, evaluate_functional, get_iteration, list_iterations, get_lineage,
                   record_summary, add_memory, get_memory, get_token_usage, search_stats, worker_status
Tools (SCF):       scf_check_eligibility, scf_submit, scf_status
Tools (final):     final_test_evaluation   (disabled unless DFTSR_ALLOW_TEST=1)

Every evaluate_functional call is recorded as one iteration in runs/iterations.sqlite together with its
accuracy (train/validation WRMSD, per-category errors, constraint results, score) and the agent's
tokens consumed since the previous iteration.
"""
from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from dftsr import tracker  # noqa: E402
from dftsr.config import RUNS_DIR, SCF_RUNS_DIR, FEATURE_CACHE_DIR, DENSITY_DIR  # noqa: E402

CAND_DIR = os.path.join(ROOT, 'candidates')
GPUS = [int(g) for g in os.environ.get('DFTSR_GPUS', '0').split(',') if g.strip() != '']
ALLOW_TEST = os.environ.get('DFTSR_ALLOW_TEST', '0') == '1'
# The MCP client's tool-call timeout is 600 s (10 min): the GPU job must finish (or fail cleanly) before that.
JOB_TIMEOUT = float(os.environ.get('DFTSR_JOB_TIMEOUT', '560'))
FIT_BUDGET = float(os.environ.get('DFTSR_FIT_BUDGET', str(max(30.0, JOB_TIMEOUT - 150))))
# SCF gate: candidate must pass all constraints and beat the ωB97M-V non-SCF validation WRMSD by this factor
SCF_GATE_FACTOR = float(os.environ.get('DFTSR_SCF_GATE', '0.97'))

mcp = FastMCP('dft-functional-search')


# ============================================================================ worker pool
class GPUWorker:
    def __init__(self, gpu):
        self.gpu = gpu
        self.proc = None
        self.conn = None
        self.busy = False
        self.ready_info = None
        self.lock = threading.Lock()

    def start(self):
        from dftsr.worker import worker_main
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(self.gpu, child), daemon=True)
        self.proc.start()
        self.conn = parent
        if not self.conn.poll(1800):
            raise RuntimeError(f'GPU {self.gpu} worker did not become ready')
        msg = self.conn.recv()
        if not msg.get('ready'):
            self.proc.kill()
            raise RuntimeError(f'GPU {self.gpu} worker failed to start: {msg.get("error")}\n{msg.get("traceback", "")}')
        self.ready_info = msg

    def alive(self):
        return self.proc is not None and self.proc.is_alive()

    def run(self, job, timeout):
        with self.lock:
            self.busy = True
            try:
                if not self.alive():
                    self.start()
                self.conn.send(job)
                if not self.conn.poll(timeout):
                    self.proc.kill()
                    self.proc = None
                    return dict(status='error', error=f'evaluation timed out after {timeout:.0f}s (worker restarted)')
                return self.conn.recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                self.proc = None
                return dict(status='error', error=f'worker crashed ({e!r}); it will be restarted on the next job. '
                                                  'Likely cause: GPU OOM or a hard crash in the candidate code.')
            finally:
                self.busy = False


class Pool:
    def __init__(self, gpus):
        self.workers = [GPUWorker(g) for g in gpus]
        self.free = queue.Queue()
        for w in self.workers:
            self.free.put(w)

    def submit(self, job, timeout=JOB_TIMEOUT):
        t0 = time.time()
        w = self.free.get()          # blocks until a GPU worker is free
        try:
            t1 = time.time()
            with w.lock:
                cold = not w.alive()
                if cold:
                    w.start()        # worker start-up (data load) counted as queue time, not job time
            queue = time.time() - t0
            res = w.run(job, timeout)
            res['queue_seconds'] = round(queue, 1)
            res['worker_cold_start_seconds'] = round(queue - (t1 - t0), 1) if cold else 0.0
            return res
        finally:
            self.free.put(w)

    def warm(self):
        """Start every GPU worker (loads train+validation caches onto each GPU) in parallel threads."""
        def _start(w):
            with w.lock:
                if not w.alive():
                    try:
                        w.start()
                    except Exception as e:  # reported via worker_status
                        w.ready_info = dict(error=repr(e))
        ts = [threading.Thread(target=_start, args=(w,), daemon=True) for w in self.workers]
        for t in ts:
            t.start()
        return ts

    def status(self):
        return [dict(gpu=w.gpu, alive=w.alive(), busy=w.busy,
                     log=(w.ready_info or {}).get('log')) for w in self.workers]


POOL = Pool(GPUS)


# ============================================================================ helpers
def _db():
    return tracker.connect()


def _j_target(con):
    v = tracker.get_meta(con, 'j_target')
    return float(v) if v else None


def _baseline_val(con):
    v = tracker.get_meta(con, 'baseline_val_wrmsd')
    return float(v) if v else None


def _slug(name):
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)[:80] or 'candidate'


def _row(con, iid):
    r = con.execute('SELECT * FROM iterations WHERE id=?', (iid,)).fetchone()
    return dict(r) if r else None


def _brief_row(r, extra=False):
    out = {k: r[k] for k in ('id', 'name', 'parent_id', 'island', 'family', 'status', 'n_params', 'train_wrmsd',
                             'val_wrmsd', 'score', 'n_violations', 'eval_seconds', 'wall_seconds', 'gap_seconds',
                             'tokens_total', 'created')}
    for k in ('train_wrmsd', 'val_wrmsd', 'score'):
        if out[k] is not None:
            out[k] = round(out[k], 4)
    if extra:
        out['plan'] = (r['plan'] or '')[:400]
        out['summary'] = (r['summary'] or '')[:400]
    return out


def _format_result(res, iid, delta, baseline):
    out = dict(iteration_id=iid, status=res.get('status'), name=res.get('name'), family=res.get('family'))
    if res.get('status') != 'ok':
        out['error'] = res.get('error')
        if res.get('traceback'):
            out['traceback_tail'] = res['traceback'][-1500:]
        out['tokens_this_iteration'] = delta
        out['timings_seconds'] = res.get('timings')
        return out
    out.update(train_wrmsd=round(res['train_wrmsd'], 4), val_wrmsd=round(res['val_wrmsd'], 4),
               score=round(res.get('score') or 0, 4), n_params=res['n_params'], n_violations=res['n_violations'])
    if baseline:
        out['val_vs_wB97MV_baseline_pct'] = round(100 * (1 - res['val_wrmsd'] / baseline), 2)
    out['validation_by_type'] = {k: dict(wrmsd=round(v['wrmsd'], 3), rmsd_kcal=round(v['rmsd'], 3), n=v['n'])
                                 for k, v in res['validation']['by_type'].items()}
    out['train_by_type'] = {k: round(v['wrmsd'], 3) for k, v in res['train']['by_type'].items()}
    out['constraints'] = {k: ({kk: vv for kk, vv in v.items() if kk in ('passed', 'value', 'threshold', 'error')}
                              if isinstance(v, dict) else v) for k, v in res['constraints'].items()}
    out['fitted_params'] = res['params']
    out['fit'] = res.get('fit')
    out['missing_reactions'] = dict(train=res['train']['n_missing'], validation=res['validation']['n_missing'])
    out['eval_seconds'] = res['eval_seconds']
    out['timings_seconds'] = res.get('timings')
    out['tokens_this_iteration'] = delta
    return out


# ============================================================================ tools: task
@mcp.tool()
def get_task_brief() -> str:
    """Return the task definition, rules, candidate-code API/template, baseline numbers and data status.
    Call this first in a new session."""
    con = _db()
    prompt = open(os.path.join(ROOT, 'prompts', 'NONSCF_SEARCH_PROMPT.md')).read()
    forms = os.path.join(ROOT, 'docs', 'FUNCTIONAL_FORMS.md')
    caches = {sp: os.path.exists(os.path.join(FEATURE_CACHE_DIR, sp, 'mols.json')) for sp in ('train', 'validation', 'test')}
    n_dens = len([f for f in os.listdir(DENSITY_DIR) if f.endswith('.h5')]) if os.path.isdir(DENSITY_DIR) else 0
    status = dict(baseline_val_wrmsd=_baseline_val(con), j_target=_j_target(con), feature_caches=caches,
                  densities_available=n_dens, gpus=GPUS, n_iterations=con.execute('SELECT COUNT(*) FROM iterations').fetchone()[0],
                  functional_forms_doc=forms, baselines_dir=os.path.join(ROOT, 'baselines'))
    return prompt + '\n\n## Live status\n```json\n' + json.dumps(status, indent=1) + '\n```'


@mcp.tool()
async def evaluate_functional(name: str, code: str, plan: str, parent_id: int | None = None, island: str | None = None,
                        n_restarts: int = 1, max_iter: int = 300, tags: str | None = None) -> str:
    """Fit and score one candidate functional (non-SCF, fixed ωB97M-V densities). Records an iteration.

    Args:
      name: short CamelCase name of the functional form.
      code: full Python source of the candidate file (one `Functional` subclass, see get_task_brief).
      plan: the blueprint: what changed relative to the parent and WHY it should help (physics rationale).
      parent_id: iteration id this candidate was derived from (None for seeds).
      island: optional population/island label for multi-lineage search.
      n_restarts: L-BFGS restarts (1 = start only from init_params; extra restarts add random perturbations).
      max_iter: L-BFGS iterations per restart.
      tags: optional comma-separated labels.
    Returns JSON: train/validation WRMSD (kcal/mol), per-category errors, constraint results, score,
    fitted parameters, tokens consumed this iteration.
    """
    con = _db()
    try:
        compile(code, name, 'exec')
    except SyntaxError as e:
        return json.dumps(dict(status='error', error=f'SyntaxError: {e}'))
    t_call = time.time()
    delta, now = tracker.token_delta(con)
    prev = con.execute("SELECT created FROM iterations WHERE COALESCE(island, '') != 'seed' "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    gap = None
    if prev and prev[0]:
        gap = round(t_call - time.mktime(time.strptime(prev[0], '%Y-%m-%d %H:%M:%S')), 1)
    cur = con.execute('INSERT INTO iterations (created, name, parent_id, island, status, plan, code, tags, '
                      'tokens_input, tokens_output, tokens_cache_read, tokens_cache_creation, tokens_total, '
                      'cum_tokens_total, session_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                      (tracker.now(), name, parent_id, island, 'running', plan, code, tags, delta['input'],
                       delta['output'], delta['cache_read'], delta['cache_creation'], delta['total'],
                       now['total'], now['session_id']))
    con.commit()
    iid = cur.lastrowid
    os.makedirs(CAND_DIR, exist_ok=True)
    path = os.path.join(CAND_DIR, f'{iid:05d}_{_slug(name)}.py')
    with open(path, 'w') as f:
        f.write(code)
    con.execute('UPDATE iterations SET code_path=? WHERE id=?', (path, iid))
    con.commit()
    opts = dict(n_restarts=max(1, min(int(n_restarts), 8)), max_iter=max(10, min(int(max_iter), 1000)),
                j_target=_j_target(con), time_limit=FIT_BUDGET)
    res = await asyncio.to_thread(POOL.submit, dict(path=path, opts=opts))
    wall = round(time.time() - t_call, 1)
    res.setdefault('timings', {})
    res['timings'].update(queue=res.get('queue_seconds'), wall_tool_call=wall)
    con = _db()
    con.execute('UPDATE iterations SET wall_seconds=?, queue_seconds=?, gap_seconds=?, timings_json=? WHERE id=?',
                (wall, res.get('queue_seconds'), gap, json.dumps(res['timings']), iid))
    con.execute('UPDATE iterations SET status=?, family=?, n_params=?, params_json=?, train_wrmsd=?, val_wrmsd=?, '
                'score=?, n_violations=?, constraints_json=?, metrics_json=?, error=?, eval_seconds=? WHERE id=?',
                (res.get('status'), res.get('family'), res.get('n_params'), json.dumps(res.get('params')),
                 res.get('train_wrmsd'), res.get('val_wrmsd'), res.get('score'), res.get('n_violations'),
                 json.dumps(res.get('constraints')),
                 json.dumps({k: res.get(k) for k in ('train', 'validation', 'fit', 'log')}),
                 res.get('error'), res.get('eval_seconds'), iid))
    if res.get('status') == 'ok' and res.get('n_violations') == 0 and _baseline_val(con) and \
            res['val_wrmsd'] < SCF_GATE_FACTOR * _baseline_val(con):
        con.execute("UPDATE iterations SET scf_status='eligible' WHERE id=?", (iid,))
    con.commit()
    return json.dumps(_format_result(res, iid, delta, _baseline_val(con)), indent=1)


@mcp.tool()
def get_iteration(iteration_id: int, include_code: bool = True) -> str:
    """Full record of one iteration: plan, code, fitted parameters, metrics, constraints, summary, tokens."""
    con = _db()
    r = _row(con, iteration_id)
    if not r:
        return json.dumps(dict(error='no such iteration'))
    if not include_code:
        r.pop('code', None)
    for k in ('params_json', 'constraints_json', 'timings_json'):
        r[k] = json.loads(r[k]) if r.get(k) else None
    m = json.loads(r.pop('metrics_json') or '{}') or {}
    r['validation_by_subset_rmsd'] = (m.get('validation') or {}).get('by_subset_rmsd')
    r['validation_by_type'] = (m.get('validation') or {}).get('by_type')
    r['train_by_type'] = (m.get('train') or {}).get('by_type')
    r['fit'] = m.get('fit')
    return json.dumps(r, indent=1, default=str)


@mcp.tool()
def list_iterations(sort_by: str = 'score', limit: int = 20, island: str | None = None,
                    status: str | None = 'ok', include_plans: bool = False) -> str:
    """List iterations. sort_by: score | val_wrmsd | train_wrmsd | id | recent. Filters: island, status."""
    con = _db()
    order = {'score': 'score DESC', 'val_wrmsd': 'val_wrmsd ASC', 'train_wrmsd': 'train_wrmsd ASC',
             'id': 'id ASC', 'recent': 'id DESC'}.get(sort_by, 'score DESC')
    q, a = 'SELECT * FROM iterations WHERE 1=1', []
    if island:
        q += ' AND island=?'
        a.append(island)
    if status:
        q += ' AND status=?'
        a.append(status)
    q += f' ORDER BY {order} LIMIT ?'
    a.append(int(limit))
    rows = [_brief_row(dict(r), include_plans) for r in con.execute(q, a).fetchall()]
    return json.dumps(rows, indent=1)


@mcp.tool()
def get_lineage(iteration_id: int) -> str:
    """Ancestor chain (parent, grandparent, ...) with plans, scores and summaries — evolutionary memory."""
    con = _db()
    chain, cur = [], iteration_id
    while cur is not None and len(chain) < 50:
        r = _row(con, cur)
        if not r:
            break
        chain.append(_brief_row(r, extra=True))
        cur = r['parent_id']
    children = [_brief_row(dict(r)) for r in con.execute('SELECT * FROM iterations WHERE parent_id=?', (iteration_id,))]
    return json.dumps(dict(ancestors=chain, children=children), indent=1)


@mcp.tool()
def record_summary(iteration_id: int, summary: str) -> str:
    """Store the summarizer's factual diagnosis for an iteration (what happened vs. intent, root cause, key
    insight, risks). Facts only — no strategy suggestions."""
    con = _db()
    con.execute('UPDATE iterations SET summary=? WHERE id=?', (summary, iteration_id))
    con.commit()
    return json.dumps(dict(ok=True))


@mcp.tool()
def add_memory(kind: str, text: str, iteration_ids: str | None = None) -> str:
    """Add a global-memory entry. kind: dead_end | insight | todo | note. iteration_ids: comma-separated."""
    con = _db()
    con.execute('INSERT INTO memory (created, kind, text, iteration_ids) VALUES (?,?,?,?)',
                (tracker.now(), kind, text, iteration_ids))
    con.commit()
    return json.dumps(dict(ok=True))


@mcp.tool()
def get_memory(kind: str | None = None, limit: int = 100) -> str:
    """Read global memory (dead ends, insights, todos, notes) accumulated across the search."""
    con = _db()
    q = 'SELECT * FROM memory' + (' WHERE kind=?' if kind else '') + ' ORDER BY id DESC LIMIT ?'
    rows = con.execute(q, ((kind, limit) if kind else (limit,))).fetchall()
    return json.dumps([dict(r) for r in rows], indent=1)


@mcp.tool()
def get_token_usage() -> str:
    """Cumulative agent token usage of this project's sessions (incl. subagents) and the amount
    consumed since the last recorded iteration."""
    con = _db()
    now = tracker.token_usage()
    base = json.loads(tracker.get_meta(con, 'token_baseline') or 'null')
    since = {k: now[k] - (base or {}).get(k, 0) for k in ('input', 'output', 'cache_read', 'cache_creation', 'total')}
    tot = con.execute('SELECT SUM(tokens_total), COUNT(*) FROM iterations').fetchone()
    return json.dumps(dict(cumulative=now, since_last_iteration=since,
                           recorded_in_iterations=tot[0], n_iterations=tot[1],
                           accounting_started=tracker.get_meta(con, 'token_baseline_time')), indent=1)


@mcp.tool()
def search_stats() -> str:
    """Search progress: iterations, success rate, best-so-far trajectory, tokens and GPU time."""
    con = _db()
    rows = [dict(r) for r in con.execute('SELECT id, status, score, val_wrmsd, n_violations, tokens_total, '
                                         'eval_seconds, wall_seconds, queue_seconds, gap_seconds '
                                         'FROM iterations ORDER BY id')]
    best, traj = None, []
    for r in rows:
        if r['status'] == 'ok' and r['score'] is not None and (best is None or r['score'] > best[1]):
            best = (r['id'], r['score'])
            traj.append(dict(id=r['id'], score=round(r['score'], 4), val_wrmsd=round(r['val_wrmsd'], 4)))
    return json.dumps(dict(n_iterations=len(rows), n_ok=sum(r['status'] == 'ok' for r in rows),
                           n_constraint_clean=sum(r['status'] == 'ok' and r['n_violations'] == 0 for r in rows),
                           best_trajectory=traj, baseline_val_wrmsd=_baseline_val(con), j_target=_j_target(con),
                           tokens_total=sum(r['tokens_total'] or 0 for r in rows),
                           gpu_seconds=round(sum(r['eval_seconds'] or 0 for r in rows), 1),
                           timing=_timing_stats(rows)), indent=1)


def _timing_stats(rows):
    def st(key):
        v = sorted(r[key] for r in rows if r.get(key) is not None)
        if not v:
            return None
        return dict(n=len(v), total=round(sum(v), 1), mean=round(sum(v) / len(v), 1),
                    median=v[len(v) // 2], max=v[-1])
    return dict(gpu_eval_seconds=st('eval_seconds'), tool_wall_seconds=st('wall_seconds'),
                queue_seconds=st('queue_seconds'), agent_gap_seconds_between_iterations=st('gap_seconds'))


@mcp.tool()
async def warm_up_workers(wait: bool = False) -> str:
    """Start the GPU workers now (each loads ~10 GB of train+validation features onto its GPU, ~10-60 s), so
    the first evaluate_functional call does not pay the start-up cost. Call once at the start of a session."""
    ts = POOL.warm()
    if wait:
        await asyncio.to_thread(lambda: [t.join() for t in ts])
    return json.dumps(dict(started=True, waited=wait, workers=POOL.status()), indent=1, default=str)


@mcp.tool()
def worker_status() -> str:
    """GPU worker pool status (which GPUs, alive/busy, data loaded)."""
    return json.dumps(dict(gpus=GPUS, workers=POOL.status()), indent=1, default=str)


# ============================================================================ tools: SCF
def _eligibility(con, iid):
    r = _row(con, iid)
    if not r:
        return False, 'no such iteration'
    base = _baseline_val(con)
    if r['status'] != 'ok':
        return False, 'iteration did not evaluate successfully'
    if r['n_violations'] != 0:
        return False, f'{r["n_violations"]} constraint violation(s); only constraint-clean functionals go to SCF'
    if base is None:
        return False, 'baseline not established (run scripts/establish_baseline.py)'
    if r['val_wrmsd'] >= SCF_GATE_FACTOR * base:
        return False, (f'val WRMSD {r["val_wrmsd"]:.3f} not below gate {SCF_GATE_FACTOR} x ωB97M-V baseline '
                       f'{base:.3f} = {SCF_GATE_FACTOR * base:.3f}')
    return True, 'eligible'


@mcp.tool()
def scf_check_eligibility(iteration_id: int) -> str:
    """Is this iteration allowed to go to full SCF evaluation? (constraint-clean and clearly better than ωB97M-V
    non-SCF on validation)."""
    ok, why = _eligibility(_db(), iteration_id)
    return json.dumps(dict(eligible=ok, reason=why, gate_factor=SCF_GATE_FACTOR))


@mcp.tool()
def scf_submit(iteration_id: int, subset: str = 'validation', max_molecules: int = 0, gpus: str = '') -> str:
    """Launch a self-consistent evaluation of an eligible functional (detached background job).

    subset: train | validation | train+validation | all | probe (a 60-molecule stratified probe for a quick
    density-driven-error check). gpus: e.g. "1,2" (default: all GPUs not used by the non-SCF pool).
    Returns a job id; poll with scf_status. Expect hours for full splits (see README time estimates).
    """
    con = _db()
    ok, why = _eligibility(con, iteration_id)
    if not ok:
        return json.dumps(dict(submitted=False, reason=why))
    r = _row(con, iteration_id)
    cur = con.execute('INSERT INTO scf_jobs (iteration_id, created, status, subset) VALUES (?,?,?,?)',
                      (iteration_id, tracker.now(), 'queued', subset))
    con.commit()
    jid = cur.lastrowid
    outdir = os.path.join(SCF_RUNS_DIR, f'job{jid:04d}_it{iteration_id:05d}')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'params.json'), 'w') as f:
        f.write(r['params_json'])
    gl = gpus or ','.join(str(g) for g in range(3) if g not in GPUS) or '0'
    log = os.path.join(outdir, 'launcher.log')
    cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'scf_evaluate.py'), '--candidate', r['code_path'],
           '--params', os.path.join(outdir, 'params.json'), '--subset', subset, '--out', outdir,
           '--gpus', gl, '--job-id', str(jid)]
    if max_molecules:
        cmd += ['--max-molecules', str(max_molecules)]
    with open(log, 'w') as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    con.execute('UPDATE scf_jobs SET status=?, log_path=? WHERE id=?', ('running', log, jid))
    con.execute("UPDATE iterations SET scf_status='running' WHERE id=?", (iteration_id,))
    con.commit()
    return json.dumps(dict(submitted=True, job_id=jid, outdir=outdir, gpus=gl, cmd=' '.join(cmd)))


@mcp.tool()
def scf_status(job_id: int) -> str:
    """Progress / results of an SCF job (molecules done, convergence failures, SCF vs non-SCF WRMSD)."""
    con = _db()
    j = con.execute('SELECT * FROM scf_jobs WHERE id=?', (job_id,)).fetchone()
    if not j:
        return json.dumps(dict(error='no such job'))
    j = dict(j)
    outdir = os.path.dirname(j['log_path']) if j['log_path'] else None
    prog = os.path.join(outdir, 'progress.json') if outdir else None
    if prog and os.path.exists(prog):
        j['progress'] = json.load(open(prog))
    return json.dumps(j, indent=1, default=str)


# ============================================================================ tools: final test
@mcp.tool()
async def final_test_evaluation(iteration_id: int) -> str:
    """Held-out TEST-set evaluation (non-SCF). Disabled during search unless DFTSR_ALLOW_TEST=1.
    Using the test set to guide the search leaks it; use only for final reporting of a few functionals."""
    if not ALLOW_TEST:
        return json.dumps(dict(error='test evaluation disabled (set DFTSR_ALLOW_TEST=1 for final reporting)'))
    con = _db()
    r = _row(con, iteration_id)
    if not r or r['status'] != 'ok':
        return json.dumps(dict(error='iteration not ok'))
    # refit-free evaluation with stored parameters: write a params override file
    path = r['code_path']
    res = await asyncio.to_thread(POOL.submit, dict(path=path, opts=dict(
        fit=False, with_test=True, params_override=json.loads(r['params_json']), j_target=_j_target(con))))
    if res.get('status') == 'ok':
        con.execute('UPDATE iterations SET test_wrmsd=? WHERE id=?', (res['test_wrmsd'], iteration_id))
        con.commit()
    keys = ('status', 'error', 'train_wrmsd', 'val_wrmsd', 'test_wrmsd', 'valtest_noRG10_wrmsd')
    out = {k: res.get(k) for k in keys}
    if 'test' in res:
        out['test_by_type'] = res['test']['by_type']
    return json.dumps(out, indent=1)


def main():
    con = _db()
    tracker.ensure_token_baseline(con)
    # Each agent connection starts a separate server process. Stay lazy by default
    # so read-only connections do not duplicate the entire GPU data pool. The
    # coordinating session explicitly calls warm_up_workers (or opts in below).
    cwd = os.path.realpath(os.getcwd())
    prewarm = os.environ.get('DFTSR_PREWARM', '0')
    if prewarm == '1' or (
            prewarm == 'auto' and cwd.startswith(os.path.realpath(ROOT))):
        POOL.warm()
    mcp.run()


if __name__ == '__main__':
    main()
