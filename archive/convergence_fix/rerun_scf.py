#!/usr/bin/env python
"""Re-run the candidate SCF of given molecules with the spin-channel density threshold lowered (xc_torch.RHO_THR 1e-10 -> 1e-14).

Standalone, project code unchanged: the threshold is patched in this process only. Same protocol as dftsr.gscdb_scf.candidate: stored
omega-B97M-V orbitals as the start, same grids/thresholds/fallbacks, exact-integral energy. Output: one json line per molecule.
  rerun_scf.py --job DIR --candidate FILE --params FILE --names a,b,c --gpu N --out FILE [--thr 1e-14] [--compare]
--compare: also report the energy difference to the stored record (for converged molecules: threshold sensitivity).
"""
import argparse, json, os, sys, time, traceback
ap=argparse.ArgumentParser()
ap.add_argument('--job',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--params',required=True)
ap.add_argument('--names',required=True); ap.add_argument('--gpu',default='0'); ap.add_argument('--out',required=True)
ap.add_argument('--thr',type=float,default=1e-14); ap.add_argument('--compare',action='store_true')
a=ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import cupy as cp
from dftsr import gscdb as G, gscdb_scf as S, xc_torch as X
from dftsr.api import load_candidate
H=G.HARTREE_TO_KCAL
X.RHO_THR=a.thr
func=load_candidate(a.candidate); params=json.load(open(a.params))
old={}
for l in open(os.path.join(a.job,'molecules.jsonl')):
    if l.strip(): r=json.loads(l); old[r['name']]=r
for name in a.names.split(','):
    t0=time.time(); out=dict(name=name,rho_thr=a.thr)
    try:
        spec=G.molecule_spec(name); mol=G.build_mol(spec)
        mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
        mf,rec=S._scf(mol,spec,unr,func,params,dm0=dm_ref)
        out.update(cand=rec,E_scf=rec['e'],converged=rec['converged'],status='ok')
        o=old[name]
        out.update(old_E_scf=o.get('E_scf'),old_converged=o.get('converged'),old_how=o['cand'].get('how'))
        if o.get('E_scf') is not None: out['dE_new_old_kcal']=(rec['e']-o['E_scf'])*H
        out['dE_scf_nonscf_old_kcal']=(rec['e']-o['E_nonscf'])*H
    except Exception as e:
        out.update(status='error',error=repr(e)[:300],tb=traceback.format_exc()[-1200:])
    out['t']=round(time.time()-t0,1)
    with open(a.out,'a') as f: f.write(json.dumps(out)+'\n')
    cp.get_default_memory_pool().free_all_blocks()
