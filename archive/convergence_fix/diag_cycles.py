import sys, json, os, time, io, re
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
name=sys.argv[1]; cand=sys.argv[2]; params=json.load(open(sys.argv[3]))
from dftsr import xc_torch as _X; _X.RHO_THR=1e-14
spec=G.molecule_spec(name); mol=G.build_mol(spec)
func=load_candidate(cand)
mf_ref,ref=S.load_reference(mol,spec)
dm_ref=mf_ref.make_rdm1()
unr=mf_ref.istype('UKS')
mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec))
mf.verbose=4; mf.max_cycle=int(sys.argv[4]) if len(sys.argv)>4 else 60
mf.stdout=sys.stdout
t=time.time(); mf.kernel(dm0=dm_ref)
print('converged',mf.converged,'E',mf.e_tot,'t',time.time()-t, 'conv_tol',mf.conv_tol,'conv_tol_grad',mf.conv_tol_grad)
