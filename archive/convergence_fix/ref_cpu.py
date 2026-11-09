#!/usr/bin/env python
"""CPU re-computation of an unconverged closed-shell ωB97M-V reference (libxc), with optional uniform field.
Reuses the helpers of runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py (exact in-core J/K, same grids/thresholds/fallbacks).
Writes the new orbitals to --out-h5 (NOT to the store) and prints the energy against Q-Chem and the stored record.
"""
import argparse, importlib.util, json, os, sys, time
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import numpy as np
spec_=importlib.util.spec_from_file_location('aecpu','/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py')
A=importlib.util.module_from_spec(spec_); spec_.loader.exec_module(A)
from dftsr import gscdb as G
ap=argparse.ArgumentParser(); ap.add_argument('--mol',required=True); ap.add_argument('--out-h5',required=True); ap.add_argument('--out-json',required=True)
a=ap.parse_args()
t0=time.time()
spec=G.molecule_spec(a.mol); mol=G.build_mol(spec,max_memory=300000)
assert spec['mult']==1 and not spec['guess_mix'] and not spec['stability']
print(a.mol,'nao',mol.nao,'field',spec['field'],flush=True)
tt=time.time(); full=mol.intor('int2e',aosym='s8')
with mol.with_range_coulomb(0.3): lr=mol.intor('int2e',aosym='s8')
print('ERIs',round(time.time()-tt,1),'s',flush=True)
mf=A.cpu_mf(mol,spec,eris=(full,lr))
if spec['field']:                               # same as G.apply_field, CPU/numpy version
    field=spec['field']; sign=G.FIELD_SIGN
    F=np.array([field.get('X',0.0),field.get('Y',0.0),field.get('Z',0.0)])
    with mol.with_common_orig((0.0,0.0,0.0)): r=mol.intor_symmetric('int1e_r',comp=3)
    v=sign*np.einsum('x,xij->ij',F,r); e_nuc_f=-sign*float(np.einsum('a,ax,x->',mol.atom_charges(),mol.atom_coords(),F))
    h0=mf.get_hcore; e0=mf.energy_nuc
    mf.get_hcore=lambda *x,**k: h0(*x,**k)+v
    mf.energy_nuc=lambda *x,**k: e0(*x,**k)+e_nuc_f
ref=A.scf(mf)
pub=G.published_energies()['wB97M-V'][a.mol]
old=None
for l in open('/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0001_it159_datasets/molecules.jsonl'):
    r=json.loads(l)
    if r['name']==a.mol: old=r['E_ref']
res=dict(name=a.mol,converged=ref['converged'],how=ref['how'],cycles=ref['cycles'],e=ref['e'],e_published=pub,
         dE_vs_published_kcal=(ref['e']-pub)*G.HARTREE_TO_KCAL,dE_vs_stored_kcal=(ref['e']-old)*G.HARTREE_TO_KCAL,t=round(time.time()-t0,1))
print(res,flush=True)
import h5py
with h5py.File(a.out_h5,'w') as f:
    f.create_dataset('mo_coeff',data=mf.mo_coeff,compression='gzip',compression_opts=1); f.create_dataset('mo_occ',data=mf.mo_occ)
    for k,v in dict(ref,guess='minao',unrestricted=False,loaded=False,cpu_fallback=True).items():
        if isinstance(v,(int,float,str,bool)): f.attrs[k]=v
json.dump(dict(res,ref=dict(ref,guess='minao',unrestricted=False,loaded=False,cpu_fallback=True)),open(a.out_json,'w'))
