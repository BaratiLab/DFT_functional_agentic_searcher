#!/usr/bin/env python
"""Self-consistent evaluation of a candidate functional on GSCDB137 (see dftsr/gscdb_scf.py for the protocol).

  python scripts/gscdb_scf.py --candidate candidates/00159_F145_r4.py --params p.json --subset probe \
         --out /home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0001_it00159 --gpus 0,1,2
  python scripts/gscdb_scf.py --subset all --out ... --gpus 0,1,2          # ωB97M-V reference only

Per molecule: ωB97M-V SCF (or stored orbitals from --ref-dir), candidate non-SCF energy on that density, candidate
SCF. Writes molecules.jsonl (resumable: finished molecules are skipped on restart), progress.json (every 30 s,
with an ETA from a power-law fit of time vs basis size), and at the end result.json + energies.csv
(Molecule_Energies.csv layout, usable with GSCDB's Analysis/analyze.ipynb).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def worker(gpu, cand, params, q, out_path, lock, ref_dir, save_ref, mem, redo=frozenset(), nonscf_only=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import cupy as cp
    from dftsr import gscdb as G, gscdb_scf as S
    from dftsr.api import load_candidate
    func = load_candidate(cand) if cand else None
    pub = G.published_energies()['wB97M-V']
    while True:
        name = q.get()
        if name is None:
            return
        t0 = time.time()
        rec = dict(name=name, gpu=gpu)
        held = 0.0
        try:
            spec = G.molecule_spec(name)
            mol = G.build_mol(spec)
            rec.update(nao=mol.nao, mult=spec['mult'], df=S.use_df(mol, spec))
            # RI tensors that do not fit on the GPU live in pinned host memory (72 GB for C60ISO, 2220 functions).
            # Workers reserve their estimate from a shared budget; three such molecules at once made systemd-oomd
            # kill the unit on 2026-10-20.
            if rec['df'] and S.df_host_gb(mol, spec) > 0:
                held = S.df_host_gb(mol, spec)
                rec['df_host_gb_est'] = round(held, 1)
                tw = time.time()
                with mem['cond']:
                    while mem['used'].value > 0 and mem['used'].value + held > S.HOST_BUDGET_GB:
                        mem['cond'].wait(60)
                    mem['used'].value += held
                rec['t_wait'] = round(time.time() - tw, 1)
                t0 += rec['t_wait']          # 't' is compute time only
            got = None if name in redo else S.load_reference(mol, spec, ref_dir)
            if got is None:
                ref_mf, ref = S.reference(mol, spec, pub.get(name))
                if save_ref:
                    S.save_reference(name, ref_mf, ref, ref_dir)   # overwrites an invalidated stored reference
                if name in redo:
                    rec['ref_redone'] = True
            else:
                ref_mf, ref = got
            rec['ref'] = ref
            rec['E_ref'] = ref['e']
            if name in pub:
                rec['E_ref_published'] = pub[name]
                rec['dE_ref_published'] = ref['e'] - pub[name]
            if func is not None and nonscf_only:
                c = S.candidate_nonscf(mol, spec, func, params, ref_mf)
                rec['cand'] = c
                rec['E_nonscf'] = c['e_nonscf']
            elif func is not None:
                _, c = S.candidate(mol, spec, func, params, ref_mf)
                rec['cand'] = c
                rec['E_nonscf'] = c['e_nonscf']
                rec['E_scf'] = c['e']
                rec['converged'] = c['converged']
            rec['status'] = 'ok'
        except Exception as e:  # noqa: BLE001
            rec.update(status='error', error=repr(e)[:500], tb=traceback.format_exc()[-2000:])
        finally:
            if held:
                with mem['cond']:
                    mem['used'].value -= held
                    mem['cond'].notify_all()
        rec['t'] = round(time.time() - t0, 1)
        with lock:
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        cp.get_default_memory_pool().free_all_blocks()


def eta_hours(recs, todo_nao, n_gpus):
    """Fit t = a * nao^b on finished molecules, apply to the remaining ones."""
    import numpy as np
    pts = [(r['nao'], r['t']) for r in recs.values() if r.get('status') == 'ok' and r.get('nao') and r['t'] > 0]
    if len(pts) < 5 or not todo_nao:
        return None, None
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    b, la = np.polyfit(x, y, 1)
    rem = float(np.sum(np.exp(la) * np.asarray(todo_nao, float) ** b)) / 3600
    return round(rem / max(1, n_gpus), 2), dict(a=float(np.exp(la)), b=float(b), n_fit=len(pts))


def main():
    from dftsr import gscdb as G
    from dftsr.gscdb_jobs import read_records, finalize, write_energies_csv
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-molecules', default='', help='comma-separated molecules handled separately on CPU')
    ap.add_argument('--candidate', default='', help='candidate .py; empty = ωB97M-V reference only')
    ap.add_argument('--params', default='')
    ap.add_argument('--label', default='candidate')
    ap.add_argument('--subset', default='probe', help='probe | all | datasets')
    ap.add_argument('--datasets', default='', help='comma-separated GSCDB dataset names (with --subset datasets)')
    ap.add_argument('--max-nbas', type=int, default=0)
    ap.add_argument('--order', default='small-first', choices=['small-first', 'large-first'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--ref-dir', default=G.GSCDB_REF_DIR)
    ap.add_argument('--no-save-ref', action='store_true')
    ap.add_argument('--job-id', type=int, default=0)
    ap.add_argument('--retry-errors', action='store_true', help='rerun molecules that failed before')
    ap.add_argument('--nonscf-only', action='store_true',
                    help='candidate non-SCF energy only (also enabled by a file <out>/NONSCF_ONLY)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.nonscf_only = a.nonscf_only or os.path.exists(os.path.join(a.out, 'NONSCF_ONLY'))
    params = json.load(open(a.params)) if a.params else None
    mols = G.select_molecules(a.subset, a.datasets or None, a.max_nbas)
    if a.order == 'large-first':
        mols = mols[::-1]
    nao = {m: G._info_json()[m]['num_basis'] for m in mols}
    out_path = os.path.join(a.out, 'molecules.jsonl')
    recs = read_records(out_path)
    done = {n for n, r in recs.items() if r.get('status') == 'ok' or (r.get('status') == 'error' and not a.retry_errors)}
    # <out>/redo_refs.txt (one molecule per line): stored ωB97M-V orbitals known to be wrong (e.g. the GPU h-shell
    # hcore bug, 2026-10-21). They are recomputed and overwritten until a record with ref_redone=True exists.
    redo_path = os.path.join(a.out, 'redo_refs.txt')
    redo = set()
    if os.path.exists(redo_path):
        redo = {s.strip() for s in open(redo_path) if s.strip() and not s.startswith('#')}
        redo = frozenset(n for n in redo if not (recs.get(n, {}).get('status') == 'ok' and recs[n].get('ref_redone')))
        done -= {n for n in redo if recs.get(n, {}).get('status') == 'ok'}
    excluded = set(a.exclude_molecules.split(','))
    todo = [m for m in mols if m not in done and m not in excluded]
    json.dump(dict(vars(a), n_molecules=len(mols), n_redo_refs=len(redo), started=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(a.out, 'run_config.json'), 'w'), indent=1)
    ctx = mp.get_context('spawn')
    q, lock = ctx.Queue(), ctx.Lock()
    mem = dict(used=ctx.Value('d', 0.0, lock=False), cond=ctx.Condition())
    for m in todo:
        q.put(m)
    gpus = [int(g) for g in a.gpus.split(',')]
    for _ in gpus:
        q.put(None)
    procs = [ctx.Process(target=worker, args=(g, a.candidate or None, params, q, out_path, lock, a.ref_dir,
                                             not a.no_save_ref, mem, frozenset(redo), a.nonscf_only)) for g in gpus]
    for p in procs:
        p.start()
    t0 = time.time()

    def progress(finished=False):
        recs = read_records(out_path)
        left = [nao[m] for m in mols if m not in recs]
        eta, fit = eta_hours(recs, left, len(gpus))
        prog = dict(job_id=a.job_id, subset=a.subset, label=a.label, n_molecules=len(mols),
                    n_done=sum(r.get('status') == 'ok' for r in recs.values() if r['name'] in nao),
                    n_errors=sum(r.get('status') == 'error' for r in recs.values() if r['name'] in nao),
                    n_ref_unconverged=sum(r.get('status') == 'ok' and not r['ref'].get('converged')
                                          for r in recs.values()),
                    n_cand_unconverged=sum(r.get('status') == 'ok' and 'cand' in r and r['cand'].get('converged') is False
                                           for r in recs.values()),
                    elapsed_h=round((time.time() - t0) / 3600, 2), eta_h=eta, cost_fit=fit, finished=finished,
                    updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(prog, open(os.path.join(a.out, 'progress.json'), 'w'), indent=1)
        return recs, prog

    while any(p.is_alive() for p in procs):
        time.sleep(30)
        progress()
    recs, prog = progress(finished=True)
    result = finalize(recs, mols, a.label)
    json.dump(result, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)
    write_energies_csv(recs, os.path.join(a.out, 'energies.csv'), a.label)
    if a.job_id:
        from dftsr.gscdb_jobs import set_status
        set_status(a.job_id, 'done', result_summary=dict(
            n_ok=result['n_ok'], n_errors=result['n_errors'], ranking=result['ranking'][:20],
            ref_vs_published_kcal=result['ref_vs_published_kcal'], density_driven_kcal=result['density_driven_kcal']))
    print(json.dumps({k: result[k] for k in ('n_molecules', 'n_ok', 'n_errors', 'ref_vs_published_kcal',
                                             'density_driven_kcal')}, indent=1))


if __name__ == '__main__':
    main()
