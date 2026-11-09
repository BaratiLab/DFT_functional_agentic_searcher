"""Iteration database (SQLite) + agent token accounting (OpenAI Codex CLI sessions).

Every candidate functional evaluated through the MCP server becomes one row of `iterations`, with its
code, plan, metrics, constraint results, score, and the LLM tokens consumed since the previous iteration
(read from the Codex CLI session logs in ~/.codex/sessions/, filtered to sessions whose cwd is this project).
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import time

from .config import DB_PATH, PROJECT_ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT,
    name TEXT,
    parent_id INTEGER,
    island TEXT,
    family TEXT,
    status TEXT,                 -- ok | error | running
    plan TEXT,                   -- natural-language blueprint (what/why)
    code TEXT,
    code_path TEXT,
    n_params INTEGER,
    params_json TEXT,            -- fitted parameters
    train_wrmsd REAL,
    val_wrmsd REAL,
    test_wrmsd REAL,             -- only filled by an explicit final/test evaluation
    score REAL,                  -- J_target / J_val * 0.9^n_violations
    n_violations INTEGER,
    constraints_json TEXT,
    metrics_json TEXT,
    error TEXT,
    eval_seconds REAL,           -- GPU evaluation time inside the worker (fit + metrics + constraints)
    wall_seconds REAL,           -- wall time of the evaluate_functional call (submission -> result)
    queue_seconds REAL,          -- time waiting for a free GPU worker (incl. worker start-up if cold)
    gap_seconds REAL,            -- wall time since the previous iteration was submitted (agent think/code time)
    timings_json TEXT,           -- per-stage breakdown (load/smoke, compile, fit, per-feval, metrics, constraints)
    tokens_input INTEGER,        -- tokens consumed since previous iteration (this project, all sessions)
    tokens_output INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_creation INTEGER,
    tokens_total INTEGER,
    cum_tokens_total INTEGER,    -- cumulative tokens of the project's transcripts at record time
    session_id TEXT,
    summary TEXT,                -- summarizer diagnosis written after evaluation
    tags TEXT,
    scf_status TEXT              -- null | eligible | queued | running | done | failed
);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT,
    kind TEXT,                   -- dead_end | insight | todo | note
    text TEXT,
    iteration_ids TEXT
);
CREATE TABLE IF NOT EXISTS scf_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER,
    created TEXT,
    status TEXT,
    subset TEXT,
    result_json TEXT,
    log_path TEXT
);
"""


NEW_COLUMNS = {'wall_seconds': 'REAL', 'queue_seconds': 'REAL', 'gap_seconds': 'REAL', 'timings_json': 'TEXT'}


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute('PRAGMA table_info(iterations)')}
    for col, typ in NEW_COLUMNS.items():          # migrate databases created before these columns existed
        if col not in have:
            con.execute(f'ALTER TABLE iterations ADD COLUMN {col} {typ}')
    con.commit()
    return con


# ----------------------------------------------------------------------------- tokens
# Token accounting for an OpenAI Codex CLI agent. Codex writes one JSONL "rollout" per session under
# $CODEX_HOME/sessions/YYYY/MM/DD/ (default ~/.codex). The first line (session_meta) holds the session's cwd;
# event_msg/token_count lines hold the session's running total (info.total_token_usage). OpenAI counts cached
# input inside input_tokens and reasoning tokens inside output_tokens; they are split out below so the totals
# have the same meaning as elsewhere in this project: total = uncached input + cache read + cache write + output.
_NONMATCHING = set()   # rollout files already known to belong to other folders (their cwd never changes)


def _sessions_root():
    return os.path.join(os.environ.get('CODEX_HOME', os.path.expanduser('~/.codex')), 'sessions')


def project_cwds():
    """Folders whose sessions are charged to the search: the project folder, the folder the MCP server was
    launched from if different, plus DFTSR_TOKEN_CWDS (':'-separated)."""
    cwds = [os.path.realpath(PROJECT_ROOT)]
    launch = os.path.realpath(os.getcwd())
    if launch not in cwds:
        cwds.append(launch)
    cwds += [os.path.realpath(d) for d in filter(None, os.environ.get('DFTSR_TOKEN_CWDS', '').split(':'))]
    return cwds


def transcript_dirs():
    return [_sessions_root()]


def _inside(path, roots):
    path = os.path.realpath(path or '')
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def token_usage(transcript_dir=None):
    """Sum token usage over all Codex sessions (incl. sub-agent sessions) whose cwd is inside this project.

    Uses the last (largest) cumulative token_count of each session. Returns totals, the most recent session
    id and the models seen in those sessions.
    """
    root = transcript_dir or _sessions_root()
    roots = project_cwds()
    excl_path = os.path.join(os.path.dirname(DB_PATH), 'excluded_sessions.txt')
    excluded = set(open(excl_path).read().split()) if os.path.exists(excl_path) else set()
    tot = dict(input=0, output=0, cache_read=0, cache_creation=0)
    reasoning, n_calls, models, sessions = 0, 0, set(), 0
    latest, latest_t = None, -1
    for fp in sorted(glob.glob(os.path.join(root, '**', 'rollout-*.jsonl'), recursive=True)):
        if fp in _NONMATCHING or any(x in fp for x in excluded):
            continue
        try:
            with open(fp) as f:
                first = json.loads(f.readline() or 'null')
                meta = first.get('payload', {}) if isinstance(first, dict) else {}
                if not (isinstance(first, dict) and first.get('type') == 'session_meta'
                        and _inside(meta.get('cwd'), roots)):
                    _NONMATCHING.add(fp)
                    continue
                best, calls = None, 0
                for line in f:
                    if '"token_count"' in line:
                        try:
                            pl = json.loads(line).get('payload', {})
                        except json.JSONDecodeError:
                            continue
                        u = (pl.get('info') or {}).get('total_token_usage')
                        if isinstance(u, dict):
                            calls += 1
                            if best is None or int(u.get('total_tokens') or 0) >= int(best.get('total_tokens') or 0):
                                best = u
                    elif '"turn_context"' in line:
                        try:
                            m = json.loads(line).get('payload', {}).get('model')
                        except json.JSONDecodeError:
                            m = None
                        if m:
                            models.add(m)
            sid = meta.get('id') or os.path.basename(fp)[:-6]
            if sid in excluded:
                continue
            mt = os.path.getmtime(fp)
            if mt > latest_t:
                latest, latest_t = sid, mt
            sessions += 1
            if best:
                inp, cached = int(best.get('input_tokens') or 0), int(best.get('cached_input_tokens') or 0)
                cw = int(best.get('cache_write_input_tokens') or 0)
                tot['input'] += max(0, inp - cached - cw)
                tot['cache_read'] += cached
                tot['cache_creation'] += cw
                tot['output'] += int(best.get('output_tokens') or 0)
                reasoning += int(best.get('reasoning_output_tokens') or 0)
                n_calls += calls
        except (OSError, ValueError):
            continue
    tot['total'] = sum(tot.values())
    tot['reasoning_output'] = reasoning   # already included in output
    tot['n_messages'] = n_calls
    tot['n_sessions'] = sessions
    tot['models'] = sorted(models)
    tot['session_id'] = latest
    tot['transcript_dirs'] = [root]
    tot['project_cwds'] = roots
    return tot


def token_delta(con):
    """Token usage since the previous recorded iteration."""
    now = token_usage()
    prev = json.loads(get_meta(con, 'token_baseline') or 'null')
    if prev is None:
        prev = dict(input=0, output=0, cache_read=0, cache_creation=0, total=0)
    delta = {k: now[k] - prev.get(k, 0) for k in ('input', 'output', 'cache_read', 'cache_creation', 'total')}
    set_meta(con, 'token_baseline', json.dumps({k: now[k] for k in delta}))
    return delta, now


def ensure_token_baseline(con, reset=False):
    """Start token accounting from 'now' (so setup work before the search is not charged to iteration 1)."""
    if reset or get_meta(con, 'token_baseline') is None:
        u = token_usage()
        set_meta(con, 'token_baseline', json.dumps({k: u[k] for k in ('input', 'output', 'cache_read',
                                                                         'cache_creation', 'total')}))
        set_meta(con, 'token_baseline_time', now())
        return u
    return None


def get_meta(con, key):
    con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    r = con.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return r[0] if r else None


def set_meta(con, key, value):
    con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    con.execute('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', (key, value))
    con.commit()


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')
