"""Iteration database (SQLite) + Claude Code token accounting.

Every candidate functional evaluated through the MCP server becomes one row of `iterations`, with its
code, plan, metrics, constraint results, score, and the LLM tokens consumed since the previous iteration
(read from the Claude Code session transcripts in ~/.claude/projects/<project-slug>/).
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
def _slug_dir(path):
    """Claude Code stores transcripts in ~/.claude/projects/<path with '/', '_', '.' replaced by '-'>."""
    import re
    return os.path.expanduser('~/.claude/projects/' + re.sub(r'[^A-Za-z0-9]', '-', path))


def _project_transcript_dir():
    return _slug_dir(PROJECT_ROOT)


def transcript_dirs():
    """Directories whose transcripts are charged to the search: the project directory, plus the directory the
    (globally registered) MCP server was launched from if different, plus DFTSR_TRANSCRIPT_DIRS (':'-separated)."""
    dirs = [_project_transcript_dir()]
    cwd_dir = _slug_dir(os.path.realpath(os.getcwd()))
    if cwd_dir not in dirs:
        dirs.append(cwd_dir)
    for d in filter(None, os.environ.get('DFTSR_TRANSCRIPT_DIRS', '').split(':')):
        dirs.append(d)
    return [d for d in dirs if os.path.isdir(d)]


def token_usage(transcript_dir=None):
    """Sum token usage over all Claude Code transcripts (main sessions + subagents) of this project.

    Assistant messages are de-duplicated by message id (a message with several content blocks is logged
    on several lines carrying the same usage). Returns totals and the most recent session id.
    """
    tdirs = [transcript_dir] if transcript_dir else transcript_dirs()
    files = []
    for tdir in tdirs:
        files += glob.glob(os.path.join(tdir, '**', '*.jsonl'), recursive=True)
    excl_path = os.path.join(os.path.dirname(DB_PATH), 'excluded_sessions.txt')
    excluded = set(open(excl_path).read().split()) if os.path.exists(excl_path) else set()
    files = sorted(f for f in set(files) if not any(x in f for x in excluded))
    seen = {}
    latest, latest_t = None, -1
    for fp in files:
        try:
            mt = os.path.getmtime(fp)
            if mt > latest_t and os.path.dirname(fp) in tdirs:
                latest, latest_t = os.path.basename(fp)[:-6], mt
            with open(fp) as f:
                for line in f:
                    if '"usage"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    m = d.get('message')
                    if not isinstance(m, dict) or not isinstance(m.get('usage'), dict):
                        continue
                    key = m.get('id') or d.get('uuid')
                    seen[key] = m['usage']
        except OSError:
            continue
    tot = dict(input=0, output=0, cache_read=0, cache_creation=0)
    for u in seen.values():
        tot['input'] += int(u.get('input_tokens') or 0)
        tot['output'] += int(u.get('output_tokens') or 0)
        tot['cache_read'] += int(u.get('cache_read_input_tokens') or 0)
        tot['cache_creation'] += int(u.get('cache_creation_input_tokens') or 0)
    tot['total'] = sum(tot.values())
    tot['n_messages'] = len(seen)
    tot['session_id'] = latest
    tot['transcript_dirs'] = tdirs
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
