import sys, json, os, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S, xc_torch as X
from dftsr.api import load_candidate
name=sys.argv[1]; cand=sys.argv[2]; params=json.load(open(sys.argv[3]))
spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
func=load_candidate(cand)
E0=None
for cfg in sys.argv[4].split(';'):
    kv=dict(x.split('=') for x in cfg.split(','))
    X.RHO_THR=float(kv.get('RHO',1e-14)); X.TAU_THR=float(kv.get('TAU',1e-14)); X.SIGMA_THR=float(kv.get('SIGMA',1e-20))
    mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec)); mf.verbose=0; mf.max_cycle=150
    t=time.time(); mf.kernel(dm0=(None if os.environ.get("GUESS")=="minao" else dm_ref))
    g=mf.get_grad(mf.mo_coeff,mf.mo_occ,mf.get_fock(dm=mf.make_rdm1()))
    E=float(mf.e_tot); E0=E if E0 is None else E0
    print(cfg,'converged',bool(mf.converged),'cycles',mf.cycles,'E %.10f'%E,'dE_kcal %.5f'%((E-E0)*627.5095),'|g| %.2e'%float(cp.linalg.norm(g)),'t %.1f'%(time.time()-t),flush=True)
