#!/usr/bin/env python
"""Merge the convergence-fix results into molecules.jsonl of jobs 0001 and 0004 (and the two replaced references into 0002, 0003).
Old records are kept in <job>/molecules_before_convergence_fix.jsonl. Each changed candidate record gets cand.convergence in
{'strict','strict_gradient','noise_limited','unconverged'} plus rho_thr (xc_torch.RHO_THR used) and the diagnostics."""
import json, os, shutil, sys, collections
R='/home/ash/Documents/dft_fucntional_sr_gold_std/runs/'; C=R+'convergence_fix/'
H=627.5094740631
def load(f): return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
def recs(job): return {r['name']:r for r in load(R+job+'/molecules.jsonl')}
# -- results of the reruns (later files override earlier ones for the same molecule)
gas22=collections.OrderedDict()
for f in ('rerun_gas22a.jsonl','rerun_gas22b.jsonl'):
    for r in load(C+f):
        if r['status']=='ok' and r['converged']: r['convergence']='strict'; gas22[r['name']]=r      # DIIS strict at rho_thr 1e-14 (7 molecules)
for f in ('rerun2_gas22a.jsonl','rerun2_gas22b.jsonl','rerun3_near.jsonl','rerun5_k.jsonl'):
    for r in load(C+f):
        if r['status']=='ok': gas22[r['name']]=r
for r in load(C+'rerun3_full.jsonl'):                                         # Pd: full chain, still unconverged
    if r['status']=='ok' and r['name'] not in gas22 or (r['status']=='ok' and gas22[r['name']].get('convergence')=='unconverged'):
        r['convergence']='unconverged'; gas22[r['name']]=r
it159={}
for r in load(C+'rerun2_it159.jsonl'):
    if r['status']=='ok': it159[r['name']]=r
# -- the two molecules with replaced references: recomputed full records
two={}
for tag,job in (('j1','job0001_it159_datasets'),('j2','job0002_it396_datasets'),('j3','job0003_GAS22_datasets'),('j4','job0004_GAS22_SCF_datasets')):
    for r in load(C+f'recomp_{tag}.jsonl'):
        if r['status']=='ok': two[(job,r['name'])]=r
# job 4 Kr-Kr 3.9: SCF from the noise-aware run, non-SCF from the non-SCF recomputation, both on the new reference
kk_scf=[r for r in load(C+'krkr_scf.jsonl') if r['status']=='ok']; kk_non=[r for r in load(C+'krkr_non.jsonl') if r['status']=='ok']
def apply(job,rerun,label):
    old=recs(job); new=collections.OrderedDict(old); changed=[]
    shutil.copy(R+job+'/molecules.jsonl',R+job+'/molecules_before_convergence_fix.jsonl')
    for n,r in rerun.items():
        o=new[n]; c=dict(r['cand']); c.update(e_nonscf=o['cand']['e_nonscf'],t_nonscf=o['cand'].get('t_nonscf'),convergence=r['convergence'],rho_thr=r['rho_thr'])
        for k in ('g_best','E_spread_5best'):
            if k in r: c[k]=r[k]
        o=dict(o); o['cand']=c; o['E_scf']=r['E_scf']; o['converged']=r['converged']; o['convergence_fix']=r['convergence']; new[n]=o; changed.append(n)
    return new,changed
out={}
new1,ch1=apply('job0001_it159_datasets',it159,'it159'); out['job0001_it159_datasets']=new1
new4,ch4=apply('job0004_GAS22_SCF_datasets',gas22,'GAS22'); out['job0004_GAS22_SCF_datasets']=new4
for job in ('job0002_it396_datasets','job0003_GAS22_datasets'):
    if job not in out:
        shutil.copy(R+job+'/molecules.jsonl',R+job+'/molecules_before_convergence_fix.jsonl'); out[job]=collections.OrderedDict(recs(job))
# replaced references (all four jobs): take the recomputed record, then re-apply the fixed SCF where a rerun exists
for (job,n),r in two.items():
    if job=='job0004_GAS22_SCF_datasets' and n=='RG10N_KrKr_3p900': continue
    out[job][n]=dict(r)
if kk_scf and kk_non:
    base=dict(two.get(('job0004_GAS22_SCF_datasets','RG10N_KrKr_3p900')) or recs('job0004_GAS22_SCF_datasets')['RG10N_KrKr_3p900'])
    nn=kk_non[-1]; s=kk_scf[-1]
    base.update(ref=nn['ref'],E_ref=nn['E_ref'],E_ref_published=nn['E_ref_published'],dE_ref_published=nn['dE_ref_published'],recomputed_with_cpu_ref=True)
    c=dict(s['cand']); c.update(e_nonscf=nn['E_nonscf'],t_nonscf=nn['cand'].get('t_nonscf'),convergence=s['convergence'],rho_thr=s['rho_thr'])
    for k in ('g_best','E_spread_5best'):
        if k in s: c[k]=s[k]
    base.update(cand=c,E_nonscf=nn['E_nonscf'],E_scf=s['E_scf'],converged=s['converged'],convergence_fix=s['convergence'],status='ok')
    out['job0004_GAS22_SCF_datasets']['RG10N_KrKr_3p900']=base
for job,d in out.items():
    with open(R+job+'/molecules.jsonl','w') as f:
        for r in d.values(): f.write(json.dumps(r)+'\n')
    n=len(d); unc=[x for x,r in d.items() if r.get('converged') is False]; ru=[x for x,r in d.items() if not r['ref'].get('converged')]
    cq=collections.Counter(r.get('convergence_fix') for r in d.values() if r.get('convergence_fix'))
    print(job,n,'cand unconverged:',unc,'| ref unconverged:',ru,'| fix classes',dict(cq))
