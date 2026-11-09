#!/usr/bin/env python
"""Recompute the candidate part of a job's record for molecules whose stored omega-B97M-V reference was replaced (CPU, converged).
Standard GPU pipeline (S.candidate / S.candidate_nonscf from the stored orbitals); writes updated full records to --out (jsonl)."""
import argparse, json, os, sys, time, traceback
ap=argparse.ArgumentParser()
ap.add_argument('--job',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--params',required=True)
ap.add_argument('--names',required=True); ap.add_argument('--gpu',default='0'); ap.add_argument('--out',required=True)
ap.add_argument('--nonscf-only',action='store_true')
a=ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
sys.path.insert(0,'/home/ash/Documents/dft_functional_sr')
import cupy as cp
from dftsr import gscdb as G, gscdb_scf as S
from dftsr.api import load_candidate
func=load_candidate(a.candidate); params=json.load(open(a.params)); pub=G.published_energies()['wB97M-V']
old={}
for l in open(os.path.join(a.job,'molecules.jsonl')):
    if l.strip(): r=json.loads(l); old[r['name']]=r
for name in a.names.split(','):
    t0=time.time(); rec=dict(old[name]); rec['recomputed_with_cpu_ref']=True
    try:
        spec=G.molecule_spec(name); mol=G.build_mol(spec)
        ref_mf,ref=S.load_reference(mol,spec)
        ref['cpu_fallback']=True
        rec['ref']=ref; rec['E_ref']=ref['e']; rec['E_ref_published']=pub[name]; rec['dE_ref_published']=ref['e']-pub[name]
        if a.nonscf_only:
            c=S.candidate_nonscf(mol,spec,func,params,ref_mf); rec['cand']=c; rec['E_nonscf']=c['e_nonscf']
        else:
            _,c=S.candidate(mol,spec,func,params,ref_mf); rec['cand']=c; rec['E_nonscf']=c['e_nonscf']; rec['E_scf']=c['e']; rec['converged']=c['converged']
        rec['status']='ok'; rec['t_recompute']=round(time.time()-t0,1)
    except Exception as e:
        rec=dict(name=name,status='error',error=repr(e)[:300],tb=traceback.format_exc()[-1500:])
    with open(a.out,'a') as f: f.write(json.dumps(rec)+'\n')
    cp.get_default_memory_pool().free_all_blocks()
