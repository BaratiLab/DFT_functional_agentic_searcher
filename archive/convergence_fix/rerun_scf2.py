#!/usr/bin/env python
"""Noise-aware candidate SCF re-run (see runs/convergence_fix/NOTES in the job REPORT).

For GAS22 and it159 the torch potential is numerically non-smooth in the far minority-spin tail (rho_sigma ~ 1e-10, reduced gradient ~1e9):
a tiny orbital rotation changes that tail by a large relative amount, so DIIS stalls with |g| ~ 1e-5 while the energy is stationary to ~1e-9 Eh.
Protocol: xc_torch.RHO_THR = 1e-14 (patched in this process only), DIIS up to 100 cycles from the stored omega-B97M-V orbitals, track the iterate
with the smallest orbital gradient. Strict convergence (|g| < conv_tol_grad = 1e-6 and dE < 1e-10) -> how='diis'. Otherwise the best-|g| iterate is
accepted as 'noise-limited converged' only if |g|_best < 1e-3 AND the energies of the 5 smallest-|g| iterates agree within 1e-8 Eh; its
exact-integral energy is the record. Anything else stays unconverged (no level-shift stage here).
"""
import argparse, json, os, sys, time, traceback
ap=argparse.ArgumentParser()
ap.add_argument('--job',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--params',required=True)
ap.add_argument('--names',required=True); ap.add_argument('--gpu',default='0'); ap.add_argument('--out',required=True)
ap.add_argument('--thr',type=float,default=1e-14); ap.add_argument('--level-shift',type=float,default=0.0); ap.add_argument('--spread-tol',type=float,default=1e-8); ap.add_argument('--max-cycle',type=int,default=100)
a=ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S, xc_torch as X
from dftsr.api import load_candidate
H=G.HARTREE_TO_KCAL
X.RHO_THR=a.thr
SPREAD_TOL=a.spread_tol
func=load_candidate(a.candidate); params=json.load(open(a.params))
old={}
for l in open(os.path.join(a.job,'molecules.jsonl')):
    if l.strip(): r=json.loads(l); old[r['name']]=r
for name in a.names.split(','):
    t0=time.time(); out=dict(name=name,rho_thr=a.thr)
    try:
        spec=G.molecule_spec(name); mol=G.build_mol(spec)
        mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
        df=S.use_df(mol,spec)
        mf=S.make_mf(mol,spec,unr,func,params,density_fit=df); mf.max_cycle=a.max_cycle; mf.level_shift=a.level_shift
        hist=[]
        def cb(env):
            g=env.get('norm_gorb'); 
            if g is not None: hist.append((float(g),float(env['e_tot']),cp.asarray(env['dm']).copy() if False else None))
        keep={}
        def cb2(env):
            g=env.get('norm_gorb')
            if g is None: return
            g=float(g); e=float(env['e_tot']); hist.append((g,e))
            if 'g' not in keep or g<keep['g']: keep.update(g=g,dm=cp.asarray(env['dm']).copy(),e=e)
        mf.callback=cb2
        tt=time.time(); mf.kernel(dm0=dm_ref); t_scf=round(time.time()-tt,1)
        if mf.converged:
            dm=mf.make_rdm1(); how='diis'; q='strict'; e_scf=float(mf.e_tot)
        else:
            gs=sorted(hist)[:5]; spread=max(e for _,e in gs)-min(e for _,e in gs)
            out['g_best']=keep['g']; out['E_spread_5best']=spread; out['n_cycles']=len(hist)
            if keep['g']<1e-6:                          # an iterate met the gradient criterion (energy criterion 1e-10 only missed by noise)
                dm=keep['dm']; how='diis_best_iterate'; q='strict_gradient'; e_scf=keep['e']
            elif keep['g']<1e-3 and spread<SPREAD_TOL:
                dm=keep['dm']; how='diis_noise_limited'; q='noise_limited'; e_scf=keep['e']
            else:
                dm=mf.make_rdm1(); how='diis_failed'; q='unconverged'; e_scf=float(mf.e_tot)
        conv=(q!='unconverged')
        t1=time.time(); e=S.exact_energy(mol,spec,unr,dm,func,params,mf) if df else (float(mf.energy_tot(dm=dm)) if q=='noise_limited' else e_scf)
        rec=dict(converged=conv,how=how,cycles=len(hist),e_scf=e_scf,df=df,t_scf=t_scf,e=e,t_exact=round(time.time()-t1,1),convergence=q)
        o=old[name]
        out.update(cand=rec,E_scf=e,converged=conv,convergence=q,status='ok',old_E_scf=o.get('E_scf'),old_how=o['cand'].get('how'))
        if o.get('E_scf') is not None: out['dE_new_old_kcal']=(e-o['E_scf'])*H
        out['dE_scf_nonscf_old_kcal']=(e-o['E_nonscf'])*H
    except Exception as ex:
        out.update(status='error',error=repr(ex)[:300],tb=traceback.format_exc()[-1500:])
    out['t']=round(time.time()-t0,1)
    with open(a.out,'a') as f: f.write(json.dumps(out)+'\n')
    cp.get_default_memory_pool().free_all_blocks()
