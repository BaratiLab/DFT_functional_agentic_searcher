import sys, json, os, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
name=sys.argv[1]; spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
rng=np.random.default_rng(0)
print(name,'unrestricted',unr,'dm shape',dm.shape)
for lab,cand,pj in (('GAS22','/home/ash/Documents/dft_functional_sr/baselines/gas22.py','job0004_GAS22_SCF_datasets'),('it159','/home/ash/Documents/dft_functional_sr/candidates/00159_F145_r4.py','job0001_it159_datasets')):
    params=json.load(open(f'/home/ash/Documents/dft_fucntional_sr_gold_std/runs/{pj}/params.json'))
    func=load_candidate(cand)
    mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec)); mf.verbose=0
    mf.grids.build(); mf.nlcgrids.build()
    v1=mf.get_veff(mol,dm); v2=mf.get_veff(mol,dm)
    n=lambda x: float(cp.linalg.norm(cp.asarray(x)))
    print(lab,'repeat diff |V1-V2| %.2e'%n(v1-v2),'|V| %.2e'%n(v1))
    p=cp.asarray(rng.standard_normal(dm.shape)); p=p+p.swapaxes(-1,-2); p/=n(p)
    for h in (1e-4,1e-6,1e-8,1e-10):
        vp=mf.get_veff(mol,dm+h*p); vm=mf.get_veff(mol,dm-h*p)
        print(lab,'h %.0e'%h,'|V(+h)-V(-h)|/(2h) %.3e'%(n(vp-vm)/(2*h)),'asym |V-V^T| %.1e'%n(vp-vp.swapaxes(-1,-2)))
