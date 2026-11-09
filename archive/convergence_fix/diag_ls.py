import sys, json, os, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S, xc_torch as X
from dftsr.api import load_candidate
name=sys.argv[1]; cand=sys.argv[2]; params=json.load(open(sys.argv[3]))
X.RHO_THR=1e-14
spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
func=load_candidate(cand)
for cfg in sys.argv[4].split(';'):
    ls,damp,diis=[float(x) for x in cfg.split(',')]
    mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec)); mf.verbose=0
    mf.max_cycle=200; mf.level_shift=ls; mf.damp=damp
    if diis==0: mf.diis=False
    hist=[]
    mf.callback=lambda env: hist.append((float(env['norm_gorb']),float(env['e_tot'])))
    t=time.time(); mf.kernel(dm0=dm_ref)
    gs=sorted(hist)[:5]; sp=max(e for _,e in gs)-min(e for _,e in gs)
    print('level_shift %.2f damp %.2f diis %d'%(ls,damp,diis),'converged',bool(mf.converged),'cycles',len(hist),'E_last %.9f'%mf.e_tot,'g_best %.1e'%gs[0][0],'spread5 %.1e'%sp,'g_last %.1e'%hist[-1][0],'t %.0f'%(time.time()-t),flush=True)
