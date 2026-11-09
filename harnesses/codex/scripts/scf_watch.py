"""Detached watcher for an SCF job (independent of any agent session). Run with nohup.

Every --interval seconds it reads <outdir>/progress.json and the scf_jobs row in runs/iterations.sqlite and
appends a heartbeat line to <final_json>.log. When the job finishes (progress.json finished / DB status done)
or its process disappears without finishing, it writes <final_json> with end time, wall hours, convergence
counts, per-split SCF / non-SCF / ωB97M-V WRMSD (scripts/scf_split_report.py) and density-driven error
statistics, then exits. It never modifies the job or the database.

Usage: nohup python scripts/scf_watch.py --job-id N --outdir DIR --final runs/scf_<name>/full_jobN_final.json \
           --start-epoch S [--interval 600] &
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def job_alive(job_id):
    r = subprocess.run(['pgrep', '-f', f'scf_evaluate.py.*--job-id {job_id}$'], capture_output=True)
    return r.returncode == 0


def db_row(job_id):
    try:
        con = sqlite3.connect(os.path.join(ROOT, 'runs', 'iterations.sqlite'), timeout=30)
        con.row_factory = sqlite3.Row
        r = con.execute('SELECT * FROM scf_jobs WHERE id=?', (job_id,)).fetchone()
        return dict(r) if r else None
    except Exception as e:
        return dict(error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', type=int, required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--final', required=True)
    ap.add_argument('--start-epoch', type=float, required=True)
    ap.add_argument('--interval', type=int, default=600)
    a = ap.parse_args()
    log = open(a.final + '.log', 'a', buffering=1)
    prog_path = os.path.join(a.outdir, 'progress.json')
    misses = 0
    while True:
        prog = {}
        try:
            prog = json.load(open(prog_path))
        except Exception:
            pass
        row = db_row(a.job_id) or {}
        alive = job_alive(a.job_id)
        finished = bool(prog.get('finished')) or row.get('status') == 'done'
        log.write(json.dumps(dict(t=time.strftime('%Y-%m-%d %H:%M:%S'), alive=alive, db_status=row.get('status'),
                                  **{k: prog.get(k) for k in ('n_done', 'n_molecules', 'n_errors',
                                                               'n_unconverged', 'elapsed_h')})) + '\n')
        misses = 0 if alive else misses + 1
        if finished or misses >= 2:   # dead on two consecutive checks without finishing
            break
        time.sleep(a.interval if alive else 60)
    end = time.time()
    out = dict(job_id=a.job_id, outdir=a.outdir, outcome='finished' if finished else 'process_died_unfinished',
               start=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(a.start_epoch)),
               end=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(end)),
               wall_hours=round((end - a.start_epoch) / 3600, 3), progress=prog, scf_jobs_row=row)
    try:
        from scf_split_report import report
        out['report'] = report(a.outdir, prog.get('subset') or row.get('subset') or 'all')
    except Exception as e:
        import traceback
        out['report_error'] = traceback.format_exc()[-2000:]
    if not finished:
        try:
            out['launcher_log_tail'] = open(os.path.join(a.outdir, 'launcher.log')).read()[-4000:]
        except Exception:
            pass
    json.dump(out, open(a.final, 'w'), indent=1, default=str)
    log.write(f'final written: {a.final} outcome={out["outcome"]}\n')


if __name__ == '__main__':
    main()
