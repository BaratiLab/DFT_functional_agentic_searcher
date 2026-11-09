"""GSCDB SCF job registry (its own SQLite file in $GSCDB_RUNS, separate from runs/iterations.sqlite) and the
reporting shared by scripts/gscdb_scf.py and the MCP gscdb_* tools."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np

from . import gscdb as G

DB_PATH = os.path.join(G.GSCDB_RUNS, 'gscdb_jobs.sqlite')


def connect():
    os.makedirs(G.GSCDB_RUNS, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, iteration_id INTEGER, label TEXT, subset TEXT, datasets TEXT,
        max_nbas INTEGER, gpus TEXT, created TEXT, status TEXT, outdir TEXT, unit TEXT, cmd TEXT,
        approval TEXT, finished TEXT, result_summary TEXT)""")
    con.commit()
    return con


def create_job(**kw):
    con = connect()
    kw.setdefault('created', time.strftime('%Y-%m-%d %H:%M:%S'))
    kw.setdefault('status', 'queued')
    cols = ','.join(kw)
    cur = con.execute(f'INSERT INTO jobs ({cols}) VALUES ({",".join("?" * len(kw))})', list(kw.values()))
    con.commit()
    return cur.lastrowid


def update_job(job_id, **kw):
    con = connect()
    con.execute(f'UPDATE jobs SET {", ".join(f"{k}=?" for k in kw)} WHERE id=?', [*kw.values(), job_id])
    con.commit()


def set_status(job_id, status, result_summary=None):
    kw = dict(status=status)
    if status in ('done', 'failed', 'stopped'):
        kw['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if result_summary is not None:
        kw['result_summary'] = json.dumps(result_summary)
    update_job(job_id, **kw)


def get_job(job_id):
    r = connect().execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    return dict(r) if r else None


def list_jobs():
    return [dict(r) for r in connect().execute('SELECT id, iteration_id, label, subset, datasets, status, created, '
                                                'finished, outdir, unit FROM jobs ORDER BY id')]


# ============================================================================ reporting
def read_records(out_path):
    recs = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('status') == 'ok' or r['name'] not in recs:
                recs[r['name']] = r
    return recs


def finalize(recs, mols, label):
    """Report payload: our ωB97M-V vs the published Q-Chem ωB97M-V energies, candidate SCF / non-SCF vs the 16
    published functionals (mean NER over the datasets complete for every method), per-set metrics."""
    mols = set(mols)
    ok = {n: r for n, r in recs.items() if r.get('status') == 'ok' and n in mols}
    tables = {'wB97M-V (this work, SCF)': {n: r['E_ref'] for n, r in ok.items()}}
    if any('E_scf' in r for r in ok.values()):
        tables[f'{label} SCF'] = {n: r['E_scf'] for n, r in ok.items() if 'E_scf' in r}
    if any('E_nonscf' in r for r in ok.values()):     # also for --nonscf-only jobs
        tables[f'{label} non-SCF (ωB97M-V density)'] = {n: r['E_nonscf'] for n, r in ok.items() if 'E_nonscf' in r}
    cmp_ = G.compare(tables)
    d = np.array([r['dE_ref_published'] for r in ok.values() if 'dE_ref_published' in r]) * G.HARTREE_TO_KCAL
    dd = np.array([abs(r['E_scf'] - r['E_nonscf']) for r in ok.values() if 'E_scf' in r]) * G.HARTREE_TO_KCAL
    worst = sorted(((abs(r.get('dE_ref_published', 0)), n) for n, r in ok.items()), reverse=True)[:15]
    return dict(
        n_molecules=len(mols), n_ok=len(ok),
        n_errors=sum(r.get('status') == 'error' for n, r in recs.items() if n in mols),
        errors=sorted(n for n, r in recs.items() if r.get('status') == 'error' and n in mols)[:50],
        n_ref_unconverged=sum(not r['ref'].get('converged') for r in ok.values()),
        n_cand_unconverged=sum('cand' in r and r['cand'].get('converged') is False for r in ok.values()),
        ref_vs_published_kcal=dict(
            n=int(d.size), mean_abs=float(np.mean(np.abs(d))) if d.size else None,
            median_abs=float(np.median(np.abs(d))) if d.size else None,
            max_abs=float(np.max(np.abs(d))) if d.size else None,
            n_above_0p1=int(np.sum(np.abs(d) > 0.1)), n_above_1=int(np.sum(np.abs(d) > 1)),
            worst=[(n, round(v * G.HARTREE_TO_KCAL, 4)) for v, n in worst]),
        density_driven_kcal=dict(mean=float(dd.mean()), median=float(np.median(dd)), max=float(dd.max()))
        if dd.size else None,
        common_datasets=cmp_['common_datasets'], ranking=cmp_['ranking'], sets=cmp_['sets'])


def write_energies_csv(recs, path, label):
    """Molecule_Energies.csv layout (rows: molecules, columns: methods), for GSCDB's analyze.ipynb."""
    cols = [('wB97M-V_thiswork', 'E_ref'), (f'{label}_SCF', 'E_scf'), (f'{label}_nonSCF', 'E_nonscf')]
    with open(path, 'w') as f:
        f.write(',' + ','.join(c for c, _ in cols) + '\n')
        for n, r in sorted(recs.items()):
            if r.get('status') == 'ok':
                f.write(n + ',' + ','.join('' if r.get(k) is None else f'{r[k]:.10f}' for _, k in cols) + '\n')


# ============================================================================ cost estimate
# Model for one molecule (ωB97M-V reference + candidate SCF, one GPU), from the 2026-10-20 validation runs:
#   t = 4 s * n_atoms * (XC grid points / (99*590))            unpruned per-atom grid work, dominates small molecules
#     + B * (nao / 1344)^2.7                                   integrals; B = 351 s with RI-JK, 1300 s exact JK
#     (x 2 when the RI tensor lives in pinned host memory: C60ISO, 2220 functions, ≈ 3200 s measured)
# Anchors: aniline dimer (1344, DF) 455 s; water-size molecules 13 s; C60ISO_1 reference alone 1614 s.
# The model is rescaled by the median measured/predicted ratio of all finished molecules in $GSCDB_RUNS.


def _model_seconds(m, info=None):
    from . import gscdb_scf as S
    info = info or G._info_json()
    spec = G.molecule_spec(m)
    n = info[m]['num_basis']

    class _M:
        nao = n
    df = S.use_df(_M, spec)
    g = spec['xc_grid'] if isinstance(spec['xc_grid'], tuple) else (99, 590)
    t = 4.0 * len(spec['atoms']) * g[0] * g[1] / (99 * 590)
    t += (351.0 if df else 1300.0) * (n / 1344) ** 2.7 * (2.0 if df and S.df_host_gb(_M, spec) > 0 else 1.0)
    return t, df


def estimate(mols, n_gpus=3, with_candidate=True):
    """GPU-hours / wall-hours for a molecule list (model above, calibrated on finished molecules). Molecules
    whose ωB97M-V orbitals are already stored only need the candidate part (~half)."""
    import glob
    info = G._info_json()
    meas, pred = [], []
    for path in glob.glob(os.path.join(G.GSCDB_RUNS, '*', 'molecules.jsonl')):
        if 'superseded' in path:
            continue
        for r in read_records(path).values():
            if (r.get('status') == 'ok' and 'cand' in r and not r['ref'].get('loaded')
                    and 'state_search' not in r['ref'] and r['name'] in info):
                meas.append(r['t'])
                pred.append(_model_seconds(r['name'], info)[0])
    # time-weighted, so the large molecules that dominate the total set the scale
    scale = sum(meas) / sum(pred) if len(meas) >= 20 and sum(pred) > 3600 else 1.0
    ratios = meas
    mol_ds = {}
    for _, ds, _, st in G.load_entries():
        for _, x in st:
            mol_ds.setdefault(x, ds)
    tot, by_ds, n_exact_big = 0.0, {}, 0
    for m in mols:
        t, df = _model_seconds(m, info)
        t *= scale
        if not with_candidate or os.path.exists(os.path.join(G.GSCDB_REF_DIR, f'{m}.h5')):
            t /= 2
        if not df and info[m]['num_basis'] > 1000:
            n_exact_big += 1
        tot += t
        by_ds[mol_ds.get(m, '?')] = by_ds.get(mol_ds.get(m, '?'), 0.0) + t
    top = sorted(by_ds.items(), key=lambda x: -x[1])[:15]
    return dict(n_molecules=len(mols), gpu_hours=round(tot / 3600, 1), wall_hours=round(tot / 3600 / n_gpus, 1),
                gpu_hours_uncalibrated=round(tot / scale / 3600, 1),
                n_gpus=n_gpus, n_exact_jk_above_1000=n_exact_big, calibration_scale=round(scale, 2),
                n_calibration_molecules=len(ratios),
                most_expensive_datasets_gpu_hours=[(d, round(t / 3600, 1)) for d, t in top],
                note='rough: the calibration (2026-10-20: scale 1.66) comes from small validation molecules '
                     '(open-shell atoms on dense grids, exact JK); the two large anchors (1344 and 2220 functions) '
                     'match the uncalibrated model within 7 %, so the truth is likely between the two numbers. '
                     'The running job reports an ETA fitted on its own timings in progress.json')
