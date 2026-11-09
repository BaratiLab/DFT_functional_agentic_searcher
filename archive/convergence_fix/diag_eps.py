import sys, json, os, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
name=sys.argv[1]; params=json.load(open(sys.argv[2])); epss=sys.argv[3].split(',')
spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
E0=None
for eps in epss:
    os.environ['GAS22_EPS']=eps
    func=load_candidate('/home/ash/Documents/dft_fucntional_sr_gold_std/runs/convergence_fix/gas22_reg.py')
    mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec))
    mf.verbose=0; mf.max_cycle=150
    e_non=float(S.exact_energy(mol,spec,unr,dm_ref,func,params,mf)) if False else None
    t=time.time(); mf.kernel(dm0=dm_ref)
    g=mf.get_grad(mf.mo_coeff,mf.mo_occ,mf.get_fock(dm=mf.make_rdm1()))
    E=float(mf.e_tot); E0=E if E0 is None else E0
    print('eps',eps,'converged',bool(mf.converged),'cycles',mf.cycles,'E %.10f'%E,'dE_vs_first_kcal %.5f'%((E-E0)*627.5095),'|g| %.2e'%float(cp.linalg.norm(g)),'t %.1f'%(time.time()-t),flush=True)
