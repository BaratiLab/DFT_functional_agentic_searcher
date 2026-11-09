#!/usr/bin/env python
"""Full self-consistent evaluation of a candidate functional on (a subset of) MGCDB84.

Same protocol as the reference densities: RI-JK SCF (def2-universal-jkfit, def2-QZVPPD, grid level 3,
SG-1 VV10 grid) started from the stored ωB97M-V orbitals, then exact-integral energy at the converged density.
Also reports the non-SCF energy of the same functional on the ωB97M-V density, so the density-driven error
(SCF - non-SCF) is measured directly.

  python scripts/scf_evaluate.py --candidate candidates/00042_X.py --params p.json --subset validation \
         --out /media/ash/PortableSSD/dft_sr/scf_runs/job1 --gpus 1,2
Resumable: rerunning with the same --out skips finished molecules.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def select(subset, max_molecules=0, available_only=False):
    from dftsr.dataset import load_reactions, molecules_for, read_xyz, resolve_geometry
    from dftsr.features import available
    R = load_reactions()
    if available_only:   # only reactions whose reference densities exist already
        R = [r for r in R if all(available(m) for _, m in r.stoich)]
    if subset == 'probe':
        # small stratified probe: up to 8 validation reactions per data type, smallest molecules first
        import collections
        by = collections.defaultdict(list)
        for r in R:
            if r.split == 'validation':
                size = sum(len(read_xyz(resolve_geometry(m))[0].split('\n')) for _, m in r.stoich)
                by[r.dtype].append((size, r))
        reacts = [r for v in by.values() for _, r in sorted(v, key=lambda x: x[0])[:8]]
    else:
        splits = {'train': ['train'], 'validation': ['validation'], 'test': ['test'],
                  'train+validation': ['train', 'validation'], 'all': ['train', 'validation', 'test']}[subset]
        reacts = [r for r in R if r.split in splits]
    mols = molecules_for(reacts)
    if max_molecules:
        mols = mols[:max_molecules]
        keep = set(mols)
        reacts = [r for r in reacts if all(m in keep for _, m in r.stoich)]
    return reacts, mols


def worker(gpu, cand, params, q, out_path, lock):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import cupy as cp
    import h5py
    import torch
    from pyscf import gto
    from dftsr.api import load_candidate, params_to_tensors, FAMILIES
    from dftsr.scf import make_ks
    from dftsr.features import read_molecule, density_from_arrays, density_path
    from dftsr.dataset import read_xyz, resolve_geometry
    from dftsr import xc_torch as X

    func = load_candidate(cand)
    fam = FAMILIES[func.family]
    while True:
        name = q.get()
        if name is None:
            return
        t0 = time.time()
        rec = dict(name=name, gpu=gpu)
        try:
            atom, charge, mult = read_xyz(resolve_geometry(name))
            mol = gto.M(atom=atom, basis='def2-qzvppd', charge=charge, spin=mult - 1, verbose=0, max_memory=60000)
            # non-SCF energy on the reference density (same formula as the screening evaluator)
            kind, d, attrs, _ = read_molecule(name)
            p64 = params_to_tensors(params, 'cuda')
            dens = density_from_arrays(kind, d, device='cuda')
            with torch.no_grad():
                e_c = float((torch.as_tensor(d['w'], device='cuda') * func.energy_density(p64, dens)).sum())
                e_r = float((torch.as_tensor(d['w'], device='cuda') * X.e_xc_b97m_family(dens)).sum())
            hf = fam['cx_sr'] * (attrs['E_xx'] - attrs['E_xx_lr']) + fam['cx_lr'] * attrs['E_xx_lr']
            hf_ref = 0.15 * (attrs['E_xx'] - attrs['E_xx_lr']) + attrs['E_xx_lr']
            rec['E_nonscf'] = float(attrs['E_tot'] + hf - hf_ref + e_c - e_r)
            # SCF from the ωB97M-V orbitals
            with h5py.File(density_path(name), 'r') as f:
                mo_c, mo_o = f['mo_coeff'][:], f['mo_occ'][:]
            mf = make_ks(mol, func, params)
            mf.conv_tol = 1e-9
            mf.max_cycle = 100
            dm0 = mf.make_rdm1(cp.asarray(mo_c), cp.asarray(mo_o))
            mf.kernel(dm0=dm0)
            if not mf.converged:
                dm1 = mf.make_rdm1()
                mf.level_shift, mf.damp, mf.max_cycle = 0.25, 0.5, 300
                mf.kernel(dm0=dm1)
            rec['converged'] = bool(mf.converged)
            rec['E_scf_df'] = float(mf.e_tot)
            mfx = make_ks(mol, func, params, density_fit=False)
            mfx.grids, mfx.nlcgrids = mf.grids, mf.nlcgrids
            rec['E_scf'] = float(mfx.energy_tot(dm=mf.make_rdm1()))
            rec['status'] = 'ok'
        except Exception as e:
            rec.update(status='error', error=repr(e), tb=traceback.format_exc()[-1500:])
        rec['t'] = round(time.time() - t0, 1)
        with lock:
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        cp.get_default_memory_pool().free_all_blocks()


def summarize(reacts, recs, key):
    import numpy as np
    from dftsr.dataset import HARTREE_TO_KCAL
    E = {n: r[key] for n, r in recs.items() if r.get('status') == 'ok' and r.get(key) is not None}
    rows = [(r, sum(c * E[m] for c, m in r.stoich) * HARTREE_TO_KCAL) for r in reacts
            if all(m in E for _, m in r.stoich)]
    if not rows:
        return None
    by = {}
    for r, v in rows:
        by.setdefault(r.dtype, []).append((r.weight, v - r.ref * HARTREE_TO_KCAL))
    w = np.array([r.weight for r, _ in rows])
    e = np.array([v - r.ref * HARTREE_TO_KCAL for r, v in rows])
    return dict(n=len(rows), wrmsd=float(np.sqrt(np.mean(w * e ** 2))),
                by_type={k: dict(n=len(v), wrmsd=float(np.sqrt(np.mean([a * b ** 2 for a, b in v]))))
                         for k, v in sorted(by.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--params', required=True)
    ap.add_argument('--subset', default='validation')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--max-molecules', type=int, default=0)
    ap.add_argument('--job-id', type=int, default=0)
    ap.add_argument('--available-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    params = json.load(open(a.params))
    reacts, mols = select(a.subset, a.max_molecules, a.available_only)
    out_path = os.path.join(a.out, 'molecules.jsonl')
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)['name'] for l in open(out_path) if '"status": "ok"' in l}
    todo = [m for m in mols if m not in done]
    ctx = mp.get_context('spawn')
    q, lock = ctx.Queue(), ctx.Lock()
    for m in todo:
        q.put(m)
    gpus = [int(g) for g in a.gpus.split(',')]
    for _ in gpus:
        q.put(None)
    procs = [ctx.Process(target=worker, args=(g, a.candidate, params, q, out_path, lock)) for g in gpus]
    for p in procs:
        p.start()
    t0 = time.time()
    prog = dict(job_id=a.job_id, subset=a.subset, n_molecules=len(mols))
    while any(p.is_alive() for p in procs):
        time.sleep(30)
        recs = {}
        if os.path.exists(out_path):
            for l in open(out_path):
                r = json.loads(l)
                recs[r['name']] = r
        prog = dict(job_id=a.job_id, subset=a.subset, n_molecules=len(mols),
                    n_done=sum(r.get('status') == 'ok' for r in recs.values()),
                    n_errors=sum(r.get('status') == 'error' for r in recs.values()),
                    n_unconverged=sum(r.get('status') == 'ok' and not r.get('converged') for r in recs.values()),
                    elapsed_h=round((time.time() - t0) / 3600, 2), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(prog, open(os.path.join(a.out, 'progress.json'), 'w'), indent=1)
    recs = {}
    for l in open(out_path):
        r = json.loads(l)
        recs[r['name']] = r
    scf, nonscf = summarize(reacts, recs, 'E_scf'), summarize(reacts, recs, 'E_nonscf')
    dd = [abs(r['E_scf'] - r['E_nonscf']) * 627.509474 for r in recs.values() if r.get('status') == 'ok']
    result = dict(subset=a.subset, scf=scf, nonscf=nonscf,
                  density_driven_kcal=dict(mean=sum(dd) / max(1, len(dd)), max=max(dd, default=None)),
                  n_molecules=len(mols), n_ok=sum(r.get('status') == 'ok' for r in recs.values()),
                  n_unconverged=sum(r.get('status') == 'ok' and not r.get('converged') for r in recs.values()))
    json.dump(result, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)
    json.dump(dict(prog, finished=True, result=result), open(os.path.join(a.out, 'progress.json'), 'w'), indent=1)
    if a.job_id:
        from dftsr import tracker
        con = tracker.connect()
        con.execute('UPDATE scf_jobs SET status=?, result_json=? WHERE id=?', ('done', json.dumps(result), a.job_id))
        it = con.execute('SELECT iteration_id FROM scf_jobs WHERE id=?', (a.job_id,)).fetchone()[0]
        con.execute("UPDATE iterations SET scf_status='done' WHERE id=?", (it,))
        con.commit()
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
