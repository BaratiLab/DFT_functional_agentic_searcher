import sys, json, os
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np, cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
name=sys.argv[1]; spec=G.molecule_spec(name); mol=G.build_mol(spec)
mf_ref,ref=S.load_reference(mol,spec); dm=mf_ref.make_rdm1(); unr=mf_ref.istype('UKS')
rng=np.random.default_rng(0); n=lambda x: float(cp.linalg.norm(cp.asarray(x)))
p=cp.asarray(rng.standard_normal(dm.shape)); p=p+p.swapaxes(-1,-2); p/=n(p)
def run(lab,mf):
    mf.verbose=0; mf.grids.build(); mf.nlcgrids.build()
    mfx=mf
    for h in (1e-6,1e-8,1e-10):
        vp=mfx.get_veff(mol,dm+h*p); vm=mfx.get_veff(mol,dm-h*p)
        print(lab,'h %.0e'%h,'|dV|=%.3e'%n(vp-vm),'ratio/(2h) %.3e'%(n(vp-vm)/(2*h)),flush=True)
run('libxc wB97M-V',S.make_mf(mol,spec,unr,None,None,density_fit=S.use_df(mol,spec)))
params=json.load(open('/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0004_GAS22_SCF_datasets/params.json'))
func=load_candidate('/home/ash/Documents/dft_functional_sr/baselines/gas22.py')
# semilocal only: candidate numint with nlc switched off (xc string without VV10)
from dftsr.scf import TorchNumInt
from gpu4pyscf import dft
mf=S.make_mf(mol,spec,unr,func,params,density_fit=S.use_df(mol,spec))
mf.do_nlc=lambda: False
run('GAS22 no-VV10 (hack)',mf)
mf=S.make_mf(mol,spec,unr,None,None,density_fit=S.use_df(mol,spec)); mf.do_nlc=lambda: False
run('libxc no-VV10 (hack)',mf)
