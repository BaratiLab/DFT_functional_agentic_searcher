import sys, json, os
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp, scipy.linalg
from pyscf.dft import numint as cnum
from dftsr import gscdb as G, gscdb_scf as S, xc_torch as X
from dftsr.api import load_candidate
from dftsr.scf import TorchNumInt
name=sys.argv[1]; cand=sys.argv[2]; params=json.load(open(sys.argv[3]))
spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm_ref=mf_ref.make_rdm1()
func=load_candidate(cand)
mf=S.make_mf(mol,spec,True,func,params,density_fit=S.use_df(mol,spec)); mf.verbose=0; mf.max_cycle=150
mf.kernel(dm0=dm_ref)
mo=cp.asnumpy(mf.mo_coeff); occ=cp.asnumpy(mf.mo_occ)
mf.grids.build(); coords=cp.asnumpy(mf.grids.coords); w=cp.asnumpy(mf.grids.weights)
ao=cnum.eval_ao(mol,coords,deriv=1)          # (4,N,nao)
tni=TorchNumInt(func,params)
def raw(mo):
    out=np.zeros((2,5,coords.shape[0]))
    for s in range(2):
        c=mo[s][:,occ[s]>0]; 
        phi=ao[0]@c; gx=[ao[k]@c for k in (1,2,3)]
        out[s,0]=(phi**2).sum(1)
        for k in range(3): out[s,1+k]=2*(phi*gx[k]).sum(1)
        out[s,4]=0.5*sum((g**2).sum(1) for g in gx)
    return out
def vxc(mo):
    r=raw(mo); e,v,_,_=tni.eval_xc_eff('wb97m-v',cp.asarray(r),1,xctype='MGGA',spin=1)
    return r,cp.asnumpy(e),cp.asnumpy(v)
r0,e0,v0=vxc(mo)
rng=np.random.default_rng(1); nmo=mo.shape[-1]
A=rng.standard_normal((2,nmo,nmo)); A=A-A.swapaxes(-1,-2)
def rot(eps): return np.stack([mo[s]@scipy.linalg.expm(eps*A[s]) for s in range(2)])
e1,e2=1e-6,1e-9
_,_,v1=vxc(rot(e1)); _,_,v2=vxc(rot(e2))
d1=(v1-v0)/e1; d2=(v2-v0)/e2
bad=np.abs(d1-d2)       # (2,5,N)
score=(bad*np.abs(w)[None,None,:]).sum((0,1))
order=np.argsort(-score)[:25]
rr=np.linalg.norm(coords,axis=1)
print('n points',coords.shape[0],'total score %.3e'%score.sum())
for i in order[:25]:
    ra,rb=r0[0,0,i],r0[1,0,i]
    s_a=(r0[0,1:4,i]**2).sum(); s_b=(r0[1,1:4,i]**2).sum()
    print('r %.3f rho_a %.2e rho_b %.2e x2_a %.2e x2_b %.2e tau_a %.2e tauW_a %.2e score %.2e |d1-d2|max %.2e'%(rr[i],ra,rb,s_a/ra**(8/3),s_b/max(rb,1e-30)**(8/3),r0[0,4,i],s_a/(8*ra),score[i],bad[:,:,i].max()))
# cumulative: fraction of total score from points by rho_b band
for lo,hi in ((0,1e-14),(1e-14,1e-12),(1e-12,1e-10),(1e-10,1e-8),(1e-8,1e-6),(1e-6,1e-3),(1e-3,1e9)):
    m=(r0[1,0]>=lo)&(r0[1,0]<hi); print('rho_b in [%.0e,%.0e): n=%d score share %.3f'%(lo,hi,m.sum(),score[m].sum()/score.sum()))
for lo,hi in ((0,1e-14),(1e-14,1e-12),(1e-12,1e-10),(1e-10,1e-8),(1e-8,1e-6),(1e-6,1e-3),(1e-3,1e9)):
    m=(r0[0,0]>=lo)&(r0[0,0]<hi); print('rho_a in [%.0e,%.0e): n=%d score share %.3f'%(lo,hi,m.sum(),score[m].sum()/score.sum()))
