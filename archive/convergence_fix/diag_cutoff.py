import sys, json, os, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
from dftsr.scf import TorchNumInt
name=sys.argv[1]; cand=sys.argv[2]; params=json.load(open(sys.argv[3]))
spec=G.molecule_spec(name); mol=G.build_mol(spec)
func=load_candidate(cand)
mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
class Cut(TorchNumInt):
    thr=0.0
    def eval_xc_eff(self,*a,**k):
        exc,vxc,x,y=super().eval_xc_eff(*a,**k)
        if self.thr>0:
            rho=a[1] if len(a)>1 else k['rho']
            rt=rho[...,0,:].sum(0) if rho.ndim==3 else rho[0]
            m=rt<self.thr
            exc[m]=0; vxc[...,m]=0
        return exc,vxc,x,y
for thr in (0.0,1e-12,1e-10,1e-8):
    mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec))
    ni=Cut(func,params); ni.thr=thr; mf._numint=ni
    mf.verbose=0; mf.max_cycle=150
    t=time.time(); mf.kernel(dm0=dm_ref)
    g=mf.get_grad(mf.mo_coeff,mf.mo_occ,mf.get_fock(dm=mf.make_rdm1()))
    print('thr',thr,'converged',bool(mf.converged),'cycles',mf.cycles,'E %.10f'%mf.e_tot,'|g| %.2e'%float(cp.linalg.norm(g)),'t %.1f'%(time.time()-t),flush=True)
