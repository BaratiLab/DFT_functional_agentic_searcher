import sys, json, os
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp, scipy.linalg
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
name=sys.argv[1]; spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm0=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
n=lambda x: float(cp.linalg.norm(cp.asarray(x)))
params=json.load(open('/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0004_GAS22_SCF_datasets/params.json'))
func=load_candidate('/home/ash/Documents/dft_functional_sr/baselines/gas22.py')
def study(lab,mf):
    mf.verbose=0; mf.max_cycle=150; mf.kernel(dm0=dm0)
    mo,occ=mf.mo_coeff,mf.mo_occ
    def grad(mo):
        dm=mf.make_rdm1(mo,occ); f=mf.get_fock(dm=dm); return cp.asarray(mf.get_grad(mo,occ,f)).ravel()
    g0=grad(mo); print(lab,'converged',bool(mf.converged),'|g0| %.2e'%n(g0),flush=True)
    rng=np.random.default_rng(1)
    nmo=mo.shape[-1]
    # random antisymmetric rotation of every spin block
    A=rng.standard_normal(mo.shape[:-2]+(nmo,nmo)); A=A-np.swapaxes(A,-1,-2)
    for eps in (1e-4,1e-6,1e-8,1e-10):
        U=np.stack([scipy.linalg.expm(eps*a) for a in A.reshape(-1,nmo,nmo)]).reshape(A.shape)
        mo2=cp.asarray(np.einsum('...ij,...jk->...ik',cp.asnumpy(mo),U))
        g1=grad(mo2)
        print(lab,'eps %.0e'%eps,'|g(eps)-g0|/eps = %.3e'%(n(g1-g0)/eps),flush=True)
study('GAS22',S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec)))
study('libxc',S.make_mf(mol,spec,unr,None,None,density_fit=S.use_df(mol,spec)))
