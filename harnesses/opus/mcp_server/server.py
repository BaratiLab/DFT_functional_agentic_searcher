#!/usr/bin/env python
"""MCP server: agentic discovery of XC functionals on MGCDB84 (non-SCF screening + gated SCF validation).

Tools (non-SCF):   get_task_brief, evaluate_functional, get_iteration, list_iterations, get_lineage,
                   record_summary, add_memory, get_memory, get_token_usage, search_stats, worker_status
Tools (SCF):       scf_check_eligibility, scf_submit, scf_status
Tools (final):     final_test_evaluation   (disabled unless DFTSR_ALLOW_TEST=1)
Tools (GSCDB137):  gscdb_info, gscdb_estimate, gscdb_scf_submit, gscdb_scf_status, gscdb_report, gscdb_stop,
                   gscdb_resume   (added 2026-10-20)
Tools (SyFES):     syfes_submit, syfes_status, syfes_report, syfes_lineage, syfes_stop, syfes_resume
                   (non-LLM regularized-evolution baseline from GAS22, added 2026-11-05)

Every evaluate_functional call is recorded as one iteration in runs/iterations.sqlite together with its
accuracy (train/validation WRMSD, per-category errors, constraint results, score) and the Claude Code
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
# Claude Code's MCP_TOOL_TIMEOUT is 600 s (10 min): the GPU job must finish (or fail cleanly) before that.
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
    """Cumulative Claude Code token usage of this project's sessions (incl. subagents) and the amount
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


# ============================================================================ tools: GSCDB137 SCF (added 2026-10-20)
# Self-consistent transfer test of a finished functional on GSCDB137 (Liang & Head-Gordon 2025), worked on in
# /home/ash/Documents/dft_fucntional_sr_gold_std. Code: dftsr/gscdb.py (dataset + analysis), dftsr/gscdb_scf.py
# (SCF protocol), dftsr/gscdb_jobs.py (job registry + reports), scripts/gscdb_scf.py (runner). Jobs run as
# `systemd-run --user` units, so they survive the Claude session and systemd-oomd kills of its cgroup.
def _gscdb_unit(job_id):
    return f'gscdb-job{job_id:04d}'


@mcp.tool()
def gscdb_info() -> str:
    """GSCDB137 overview: datasets/entries/molecules, basis-size distribution, the validation probe, stored ωB97M-V
    reference orbitals, protocol notes and the job list. Read this first in a GSCDB session."""
    import numpy as np
    from dftsr import gscdb as G, gscdb_jobs as J
    ents = G.load_entries()
    mols = G.molecules_for()
    info = G._info_json()
    nb = np.array([info[m]['num_basis'] for m in mols])
    by_type = {}
    for _, ds, _, _ in ents:
        t = G.dataset_table().get(ds, {}).get('datatype', '?')
        by_type.setdefault(t, set()).add(ds)
    n_ref = len([f for f in os.listdir(G.GSCDB_REF_DIR) if f.endswith('.h5')]) if os.path.isdir(G.GSCDB_REF_DIR) else 0
    return json.dumps(dict(
        root=G.GSCDB_ROOT, runs=G.GSCDB_RUNS, ref_dir=G.GSCDB_REF_DIR, notes=os.path.join(G.GSCDB_WORK, 'GSCDB_NOTES.md'),
        n_datasets=len({e[1] for e in ents}), n_entries=len(ents), n_molecules=len(mols),
        datasets_by_type={t: sorted(v) for t, v in sorted(by_type.items())},
        nbas_percentiles={p: int(np.percentile(nb, p)) for p in (50, 90, 99, 100)},
        n_molecules_above={t: int((nb > t).sum()) for t in (1000, 2000, 3000, 4000)},
        probe=G.probe_molecules(), n_reference_orbitals_stored=n_ref, jobs=J.list_jobs(),
        protocol=('per molecule: ωB97M-V SCF (GSCDB basis/ECP/field/grids from the Q-Chem input, RI-JK SCF + '
                  'exact-integral energy; stored and reused) -> candidate non-SCF on that density -> candidate SCF '
                  'from the ωB97M-V orbitals. Metric: GSCDB mean NER (analyze.ipynb port, verified to 2e-9 against '
                  'the published table) vs 16 published functionals.')), indent=1)


@mcp.tool()
def gscdb_estimate(subset: str = 'all', datasets: str = '', max_nbas: int = 0, n_gpus: int = 3) -> str:
    """Cost estimate (GPU-hours, wall-hours) of a GSCDB job before asking the user to approve it: power-law fits of
    time vs basis size on every finished GSCDB molecule (DF and exact-JK separately), measured anchors otherwise.
    Lists the most expensive datasets, so a max_nbas cap or dataset selection can be proposed."""
    from dftsr import gscdb as G, gscdb_jobs as J
    try:
        mols = G.select_molecules(subset, datasets or None, max_nbas)
    except ValueError as e:
        return json.dumps(dict(error=str(e)))
    ents = G.load_entries()
    keep = set(mols)
    complete = sorted({ds for _, ds, _, _ in ents} - {ds for _, ds, _, st in ents if any(m not in keep for _, m in st)})
    out = J.estimate(mols, n_gpus=n_gpus)
    out['n_datasets_complete'] = len(complete)
    out['datasets_incomplete'] = sorted({ds for _, ds, _, _ in ents} - set(complete)) if subset != 'probe' else 'n/a'
    return json.dumps(out, indent=1)


@mcp.tool()
def gscdb_scf_submit(iteration_id: int, user_approval: str, subset: str = 'probe', datasets: str = '',
                     max_nbas: int = 0, gpus: str = '0,1,2', order: str = 'small-first') -> str:
    """Launch a GSCDB137 SCF job (detached systemd unit). ONLY call this after the user has explicitly approved this
    exact run in the current conversation; pass user_approval="it<iteration_id>" (e.g. "it159"), or
    iteration_id=0 with user_approval="reference" for an ωB97M-V-reference-only job.

    subset: probe (~40 molecules covering every special case; run this first) | all (13,907 molecules, days) |
    datasets (comma-separated GSCDB dataset names in `datasets`). max_nbas skips molecules above that basis size.
    Stored ωB97M-V orbitals are reused, so later candidates cost one SCF per molecule. Poll with gscdb_scf_status.
    """
    from dftsr import gscdb as G, gscdb_jobs as J
    want = 'reference' if iteration_id == 0 else f'it{iteration_id}'
    if user_approval.strip().lower() != want:
        return json.dumps(dict(submitted=False, reason=f'user_approval must be "{want}" and the user must have '
                                                       'explicitly approved this run'))
    if subset not in ('probe', 'all', 'datasets'):
        return json.dumps(dict(submitted=False, reason='subset must be probe | all | datasets'))
    try:
        mols = G.select_molecules(subset, datasets or None, max_nbas)
    except ValueError as e:
        return json.dumps(dict(submitted=False, reason=str(e)))
    cand, params_json, label = '', None, 'wB97M-V_reference'
    if iteration_id:
        r = _row(_db(), iteration_id)
        if not r or r['status'] != 'ok':
            return json.dumps(dict(submitted=False, reason='no successful iteration with that id'))
        cand, params_json, label = r['code_path'], r['params_json'], f'it{iteration_id}'
    running = [j for j in J.list_jobs() if j['status'] == 'running']
    jid = J.create_job(iteration_id=iteration_id, label=label, subset=subset, datasets=datasets, max_nbas=max_nbas,
                       gpus=gpus, approval=user_approval)
    outdir = os.path.join(G.GSCDB_RUNS, f'job{jid:04d}_{label}_{subset}')
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'gscdb_scf.py'), '--subset', subset, '--out', outdir,
           '--gpus', gpus, '--job-id', str(jid), '--label', label, '--order', order]
    if cand:
        with open(os.path.join(outdir, 'params.json'), 'w') as f:
            f.write(params_json)
        cmd += ['--candidate', cand, '--params', os.path.join(outdir, 'params.json')]
    if datasets:
        cmd += ['--datasets', datasets]
    if max_nbas:
        cmd += ['--max-nbas', str(max_nbas)]
    log = os.path.join(outdir, 'launcher.log')
    unit = _gscdb_unit(jid)
    env = [f'--setenv={k}={os.environ[k]}' for k in ('GSCDB_WORK', 'GSCDB_ROOT', 'GSCDB_RUNS', 'GSCDB_REF_DIR',
                                                      'GSCDB_DF_MAX_NAO') if k in os.environ]
    sd = ['systemd-run', '--user', f'--unit={unit}', '--collect', f'--property=WorkingDirectory={ROOT}',
          f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', *env, *cmd]
    p = subprocess.run(sd, capture_output=True, text=True)
    if p.returncode != 0:
        J.set_status(jid, 'failed')
        return json.dumps(dict(submitted=False, reason=f'systemd-run failed: {p.stderr.strip()}'))
    J.update_job(jid, status='running', outdir=outdir, unit=unit, cmd=' '.join(cmd))
    return json.dumps(dict(submitted=True, job_id=jid, unit=unit, outdir=outdir, n_molecules=len(mols), gpus=gpus,
                           other_running_jobs=[j['id'] for j in running], cmd=' '.join(cmd)), indent=1)


@mcp.tool()
def gscdb_scf_status(job_id: int) -> str:
    """Progress of a GSCDB job: molecules done/errors/unconverged, elapsed and ETA (power-law fit of time vs basis
    size), systemd unit state, and the summary once finished."""
    from dftsr import gscdb_jobs as J
    j = J.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    if j.get('unit'):
        p = subprocess.run(['systemctl', '--user', 'is-active', j['unit']], capture_output=True, text=True)
        j['unit_state'] = p.stdout.strip()
        if j['status'] == 'running' and j['unit_state'] not in ('active', 'activating'):
            prog = os.path.join(j['outdir'], 'progress.json')
            fin = os.path.exists(prog) and json.load(open(prog)).get('finished')
            if not fin:
                J.set_status(job_id, 'failed')
                j['status'] = 'failed (unit exited before finishing; see launcher.log)'
    if j.get('outdir') and os.path.exists(os.path.join(j['outdir'], 'progress.json')):
        j['progress'] = json.load(open(os.path.join(j['outdir'], 'progress.json')))
    if j.get('result_summary'):
        j['result_summary'] = json.loads(j['result_summary'])
    return json.dumps(j, indent=1, default=str)


@mcp.tool()
def gscdb_report(job_id: int = 0, outdir: str = '', top: int = 20, per_set: bool = False) -> str:
    """GSCDB results so far (works on running jobs): our ωB97M-V vs the published Q-Chem ωB97M-V energies, candidate
    SCF / non-SCF mean NER ranked against the 16 published functionals over the datasets complete for every method,
    per-type mean NER, density-driven error. Pass job_id, or outdir for a run started outside the registry."""
    from dftsr import gscdb as G, gscdb_jobs as J
    if job_id:
        j = J.get_job(job_id)
        if not j:
            return json.dumps(dict(error='no such job'))
        outdir, label = j['outdir'], j['label']
    else:
        label = json.load(open(os.path.join(outdir, 'run_config.json'))).get('label', 'candidate')
    cfg = json.load(open(os.path.join(outdir, 'run_config.json')))
    mols = G.select_molecules(cfg['subset'], cfg.get('datasets') or None, cfg.get('max_nbas') or 0)
    res = J.finalize(J.read_records(os.path.join(outdir, 'molecules.jsonl')), mols, label)
    ours = [r['method'] for r in res['ranking'] if '(published)' not in r['method']]
    out = {k: res[k] for k in ('n_molecules', 'n_ok', 'n_errors', 'errors', 'n_ref_unconverged', 'n_cand_unconverged',
                               'ref_vs_published_kcal', 'density_driven_kcal')}
    out['n_common_datasets'] = len(res['common_datasets'] or [])
    out['ranking'] = [dict(rank=i + 1, method=r['method'], mean_ner=r.get('mean_ner'))
                      for i, r in enumerate(res['ranking'][:top])]
    out['ours_by_type'] = {r['method']: r.get('mean_ner_by_type') for r in res['ranking'] if r['method'] in ours}
    if per_set:
        out['sets'] = {m: res['sets'][m] for m in ours}
    return json.dumps(out, indent=1, default=str)


@mcp.tool()
def gscdb_stop(job_id: int, user_approval: str) -> str:
    """Stop a running GSCDB job (finished molecules are kept; resubmitting the same command resumes). Only with the
    user's explicit go-ahead: pass user_approval="stop <job_id>"."""
    from dftsr import gscdb_jobs as J
    j = J.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    if user_approval.strip().lower() != f'stop {job_id}':
        return json.dumps(dict(stopped=False, reason=f'user_approval must be "stop {job_id}"'))
    p = subprocess.run(['systemctl', '--user', 'stop', j['unit']], capture_output=True, text=True)
    J.set_status(job_id, 'stopped')
    return json.dumps(dict(stopped=p.returncode == 0, stderr=p.stderr.strip(), resume='gscdb_resume(job_id)'))


@mcp.tool()
def gscdb_resume(job_id: int, user_approval: str, retry_errors: bool = False) -> str:
    """Relaunch a stopped/failed GSCDB job in its own output directory; finished molecules are skipped. Same
    approval rule as gscdb_scf_submit (user_approval="it<iteration_id>", or "reference")."""
    from dftsr import gscdb_jobs as J
    j = J.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    want = 'reference' if not j['iteration_id'] else f'it{j["iteration_id"]}'
    if user_approval.strip().lower() != want:
        return json.dumps(dict(resumed=False, reason=f'user_approval must be "{want}"'))
    p = subprocess.run(['systemctl', '--user', 'is-active', j['unit']], capture_output=True, text=True)
    if p.stdout.strip() in ('active', 'activating'):
        return json.dumps(dict(resumed=False, reason='job is still running'))
    cmd = j['cmd'].split() + (['--retry-errors'] if retry_errors else [])
    log = os.path.join(j['outdir'], 'launcher.log')
    env = [f'--setenv={k}={os.environ[k]}' for k in ('GSCDB_WORK', 'GSCDB_ROOT', 'GSCDB_RUNS', 'GSCDB_REF_DIR',
                                                      'GSCDB_DF_MAX_NAO') if k in os.environ]
    sd = ['systemd-run', '--user', f'--unit={j["unit"]}', '--collect', f'--property=WorkingDirectory={ROOT}',
          f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', *env, *cmd]
    p = subprocess.run(sd, capture_output=True, text=True)
    if p.returncode != 0:
        return json.dumps(dict(resumed=False, reason=f'systemd-run failed: {p.stderr.strip()}'))
    J.update_job(job_id, status='running', finished=None)
    return json.dumps(dict(resumed=True, job_id=job_id, unit=j['unit'], outdir=j['outdir']))


# ============================================================================ tools: SyFES non-LLM baseline (2026-11-05)
# Regularized evolution of symbolic forms (Ma et al. 2022) from GAS22 on the same non-SCF data: dftsr/syfes.py
# (forms, mutations, fingerprints), dftsr/syfes_gpu.py (batched GPU CMA-ES), scripts/syfes_run.py (population
# server + one GPU worker per GPU), scripts/syfes_recheck.py (float64 re-check + constraints of the best forms),
# dftsr/syfes_jobs.py (registry runs/syfes/jobs.json + summaries). Jobs run as detached `systemd-run --user` units;
# no tool blocks longer than SYFES_MAX_WAIT seconds, so nothing hits Claude Code's 600 s MCP timeout.
SYFES_MAX_WAIT = 540


def _syfes_unit(job_id):
    return f'syfes-job{job_id:04d}'


def _syfes_state(j):
    from dftsr import syfes_jobs as SJ
    out = dict(j)
    if j.get('unit'):
        p = subprocess.run(['systemctl', '--user', 'is-active', j['unit']], capture_output=True, text=True)
        out['unit_state'] = p.stdout.strip()
    prog = os.path.join(j.get('outdir', ''), 'progress.json')
    if os.path.exists(prog):
        try:
            out['progress'] = json.load(open(prog))
        except json.JSONDecodeError:
            pass
    fin = (out.get('progress') or {}).get('finished')
    if j['status'] == 'running' and out.get('unit_state') not in ('active', 'activating'):
        new = 'finished' if fin else 'failed'
        SJ.update_job(j['id'], status=new)
        out['status'] = new if fin else 'failed (unit exited before finishing; see launcher.log)'
    log = os.path.join(j.get('outdir', ''), 'launcher.log')
    if os.path.exists(log):
        with open(log, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3000))
            out['log_tail'] = f.read().decode(errors='replace').splitlines()[-8:]
    return out


async def _syfes_wait(job_id, wait_seconds):
    """Idle (non-blocking for the server) until the job leaves 'running' or wait_seconds pass (<= 540 s)."""
    from dftsr import syfes_jobs as SJ
    t_end = time.time() + max(0.0, min(float(wait_seconds), SYFES_MAX_WAIT))
    while True:
        st = _syfes_state(SJ.get_job(job_id))
        if not str(st['status']).startswith('running') or time.time() >= t_end:
            return st
        await asyncio.sleep(min(20.0, max(1.0, t_end - time.time())))


@mcp.tool()
async def syfes_submit(n_forms: int = 400, gpus: str = '0,1,2', seed: int = 0, population: int = 100,
                       tournament: int = 25, trials: int = 5, label: str = '', wait_seconds: int = 0) -> str:
    """Launch a SyFES (non-LLM) regularized-evolution run from GAS22 on the MGCDB84 non-SCF data, as a detached
    systemd unit, and return immediately (or after idling up to wait_seconds <= 540 s, reporting progress).

    Protocol (Ma et al. 2022 / SyFES code): population 100, tournament 25, mutation probability 0.9, one mutation of
    one enhancement factor per child (SI instruction pool), functional-equivalence fingerprints, CMA-ES fit of every
    new form on train (5 restarts from N(0,1), bounds [-10,10], tolfun 1e-5, max 1e5 evaluations), fitness =
    validation WRMSD. Stops after n_forms TRAINED forms. Poll with syfes_status(job_id, wait_seconds=...)."""
    from dftsr import syfes_jobs as SJ
    running = [j for j in SJ.list_jobs() if j['status'] == 'running']
    clash = [j['id'] for j in running if set(j['gpus'].split(',')) & set(gpus.split(','))]
    if clash:
        return json.dumps(dict(submitted=False, reason=f'job(s) {clash} already running on these GPUs'))
    jid = SJ.create_job(n_forms=n_forms, gpus=gpus, seed=seed, population=population, tournament=tournament,
                        trials=trials, label=label)
    outdir = os.path.join(SJ.SYFES_DIR, f'job{jid:04d}' + (f'_{_slug(label)}' if label else ''))
    os.makedirs(outdir, exist_ok=True)
    cmd = ['bash', os.path.join(ROOT, 'scripts', 'syfes_job.sh'), outdir, str(n_forms), gpus, str(seed),
           str(population), str(tournament), str(trials)]
    log = os.path.join(outdir, 'launcher.log')
    unit = _syfes_unit(jid)
    sd = ['systemd-run', '--user', f'--unit={unit}', '--collect', f'--property=WorkingDirectory={ROOT}',
          f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
          f'--setenv=PYTHON={sys.executable}', *cmd]
    p = subprocess.run(sd, capture_output=True, text=True)
    if p.returncode != 0:
        SJ.update_job(jid, status='failed')
        return json.dumps(dict(submitted=False, reason=f'systemd-run failed: {p.stderr.strip()}'))
    SJ.update_job(jid, status='running', outdir=outdir, unit=unit, cmd=' '.join(cmd))
    out = dict(submitted=True, job_id=jid, unit=unit, outdir=outdir)
    if wait_seconds:
        out['status'] = await _syfes_wait(jid, wait_seconds)
    return json.dumps(out, indent=1, default=str)


@mcp.tool()
async def syfes_status(job_id: int = 0, wait_seconds: int = 0) -> str:
    """Progress of a SyFES run (trained forms, equivalents skipped, best validation, ETA, unit state, log tail).
    wait_seconds (<= 540): idle until the job finishes or the time passes, then report — use this to wait on a
    run without hitting the MCP timeout. job_id=0 lists all SyFES jobs."""
    from dftsr import syfes_jobs as SJ
    if not job_id:
        return json.dumps([_syfes_state(j) for j in SJ.list_jobs()], indent=1, default=str)
    if not SJ.get_job(job_id):
        return json.dumps(dict(error='no such job'))
    return json.dumps(await _syfes_wait(job_id, wait_seconds), indent=1, default=str)


@mcp.tool()
def syfes_report(job_id: int, top: int = 10, forms: bool = False, symbolic: bool = False) -> str:
    """Results of a SyFES run (works while running): counts, best-validation trajectory, top forms by validation
    (unique fingerprints), timing; plus the float64 re-check with constraint tests (recheck.json) once the run has
    finished. forms=True adds instruction lists + parameters; symbolic=True adds sympy expressions."""
    from dftsr import syfes_jobs as SJ
    j = SJ.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    out = SJ.summarize(j['outdir'], top=top, with_forms=forms, symbolic=symbolic)
    rep = os.path.join(j['outdir'], 'REPORT.md')
    out['report_md'] = rep if os.path.exists(rep) else None
    return json.dumps(out, indent=1, default=str)


@mcp.tool()
def syfes_lineage(job_id: int, form_id: int) -> str:
    """Ancestry of one SyFES form back to the GAS22 seed: mutation applied at each step and validation WRMSD."""
    from dftsr import syfes_jobs as SJ
    j = SJ.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    return json.dumps(SJ.lineage(j['outdir'], form_id), indent=1, default=str)


@mcp.tool()
def syfes_stop(job_id: int) -> str:
    """Stop a running SyFES job (history is kept; syfes_resume continues it from history.jsonl)."""
    from dftsr import syfes_jobs as SJ
    j = SJ.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    p = subprocess.run(['systemctl', '--user', 'stop', j['unit']], capture_output=True, text=True)
    SJ.update_job(job_id, status='stopped')
    return json.dumps(dict(stopped=p.returncode == 0, stderr=p.stderr.strip()))


@mcp.tool()
async def syfes_resume(job_id: int, wait_seconds: int = 0) -> str:
    """Relaunch a stopped/failed SyFES job in its own directory; it replays history.jsonl (population, fingerprints,
    counters) and re-dispatches forms that were in flight."""
    from dftsr import syfes_jobs as SJ
    j = SJ.get_job(job_id)
    if not j:
        return json.dumps(dict(error='no such job'))
    p = subprocess.run(['systemctl', '--user', 'is-active', j['unit']], capture_output=True, text=True)
    if p.stdout.strip() in ('active', 'activating'):
        return json.dumps(dict(resumed=False, reason='job is still running'))
    log = os.path.join(j['outdir'], 'launcher.log')
    sd = ['systemd-run', '--user', f'--unit={j["unit"]}', '--collect', f'--property=WorkingDirectory={ROOT}',
          f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
          f'--setenv=PYTHON={sys.executable}', *j['cmd'].split()]
    p = subprocess.run(sd, capture_output=True, text=True)
    if p.returncode != 0:
        return json.dumps(dict(resumed=False, reason=f'systemd-run failed: {p.stderr.strip()}'))
    SJ.update_job(job_id, status='running')
    out = dict(resumed=True, job_id=job_id, unit=j['unit'])
    if wait_seconds:
        out['status'] = await _syfes_wait(job_id, wait_seconds)
    return json.dumps(out, indent=1, default=str)


def main():
    con = _db()
    tracker.ensure_token_baseline(con)
    # The server is registered globally (user scope). Only pre-load GPU data when Claude Code runs inside the
    # project directory (or DFTSR_PREWARM=1); elsewhere the workers stay lazy and use no GPU memory.
    cwd = os.path.realpath(os.getcwd())
    if os.environ.get('DFTSR_PREWARM', 'auto') == '1' or (
            os.environ.get('DFTSR_PREWARM', 'auto') == 'auto' and cwd.startswith(os.path.realpath(ROOT))):
        POOL.warm()
    mcp.run()


if __name__ == '__main__':
    main()
