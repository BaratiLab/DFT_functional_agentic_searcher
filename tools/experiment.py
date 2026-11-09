#!/usr/bin/env python3
"""Frozen-parameter reader evaluations. Never fits or inserts search iterations."""
import argparse, json, math, os, subprocess, sys
from pathlib import Path
from runtime import ROOT, configure

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset',choices=['mgcdb','gscdb'])
    ap.add_argument('--scope',choices=['paper','all'],default='paper',help='GSCDB36 paper scope or unreported full GSCDB extension')
    ap.add_argument('--method',required=True,help='O159, O178, O396, C356, GAS22, wB97M-V, C2, ...')
    ap.add_argument('--mode',required=True,choices=['cached','nonscf','scf'])
    ap.add_argument('--gpus',default='0',help='one GPU for non-SCF/cached; comma-separated for SCF/GSCDB')
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--retry-errors',action='store_true',help='retry failed GSCDB molecular records')
    ap.add_argument('--include-test',action='store_true',help='frozen reporting only; never use to select/refit')
    a=ap.parse_args()
    if a.dataset=='mgcdb' and a.scope!='paper':ap.error('--scope all is only for a GSCDB extension')
    if a.dataset=='gscdb' and a.mode=='cached':ap.error('GSCDB has no search feature cache; choose nonscf or scf')
    if a.mode in ['cached','nonscf'] and a.dataset=='mgcdb' and ',' in a.gpus:ap.error('select one GPU for this mode')
    harness='codex' if a.method.startswith('C') else 'opus'
    # GSCDB implementation lives in Opus; candidate API is shared.
    h=configure('opus' if a.dataset=='gscdb' else harness)
    candidate=ROOT/'functionals'/(a.method+'.py');paramfile=candidate.with_suffix('.params.json')
    if not candidate.exists() or not paramfile.exists():ap.error('unknown frozen method')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    import hashlib
    identity=dict(dataset=a.dataset,method=a.method,mode=a.mode,scope=a.scope,include_test=a.include_test,
                  source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  params_sha256=hashlib.sha256(paramfile.read_bytes()).hexdigest())
    guard=a.out/'experiment.json'
    if guard.exists() and json.loads(guard.read_text())!=identity:ap.error('output belongs to a different experiment; use a new --out')
    guard.write_text(json.dumps(identity,indent=2)+'\n')
    def run(script,*args):subprocess.run([sys.executable,str(h/'scripts'/script),*map(str,args)],check=True)
    params=json.loads(paramfile.read_text())
    if a.dataset=='gscdb':
        ds=','.join((ROOT/'datasets/gscdb36.txt').read_text().split()) if a.scope=='paper' else ''
        subset='datasets' if a.scope=='paper' else 'all'
        run('gscdb_scf.py','--candidate',candidate,'--params',paramfile,'--label',a.method,
            '--subset',subset,'--datasets',ds,'--exclude-molecules','AE11_Rn','--gpus',a.gpus,'--out',a.out,
            *(['--nonscf-only'] if a.mode=='nonscf' else []),*(['--retry-errors'] if a.retry_errors else []))
        # Never send AE11_Rn through the defective GPU4PySCF 1.8.1 quadrature.
        rn=a.out/'ae11_cpu_fallback_record.json'
        if not rn.exists():
            Path(os.environ['GSCDB_REF_DIR']).mkdir(parents=True,exist_ok=True)
            run('ae11_cpu_fallback.py','--mol','AE11_Rn','--out',a.out,'--candidate',candidate,
                '--params',paramfile,'--label',a.method,'--write',*(['--nonscf-only'] if a.mode=='nonscf' else []))
        from dftsr import gscdb as G
        from dftsr.gscdb_jobs import read_records,finalize,write_energies_csv
        recs=read_records(str(a.out/'molecules.jsonl'));recs['AE11_Rn']=json.loads(rn.read_text())
        with (a.out/'molecules.jsonl').open('a') as f:f.write(json.dumps(recs['AE11_Rn'])+'\n')
        mols=G.select_molecules(subset,ds or None)
        result=finalize(recs,mols,a.method)
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        write_energies_csv(recs,str(a.out/'energies.csv'),a.method)
        if result['n_ok']!=len(mols) or result['n_errors']:raise RuntimeError('Incomplete GSCDB campaign; inspect result.json')
        print(json.dumps({k:result[k] for k in ['n_molecules','n_ok','n_errors','ranking']},indent=2));return
    if a.mode=='scf':
        run('scf_evaluate.py','--candidate',candidate,'--params',paramfile,'--subset',
            'all' if a.include_test else 'train+validation','--gpus',a.gpus,'--out',a.out)
        run('scf_split_report.py',a.out,'--json',a.out/'split_report.json')
        result=json.loads((a.out/'result.json').read_text())
        if result['n_ok']!=result['n_molecules']:raise RuntimeError('Incomplete MGCDB SCF campaign; inspect result.json')
        return
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpus
    splits=['train','validation']+(['test'] if a.include_test else [])
    if a.mode=='cached':
        from dftsr.worker import Workspace,evaluate
        ws=Workspace('cuda:0',splits=tuple(splits))
        if not ws.constraints.ok:raise RuntimeError('Constraint data missing; run compute_constraint_data.py first')
        if any(rs.n_missing for rs in ws.rs.values()):raise RuntimeError('Incomplete cache; refusing partial benchmark')
        result=evaluate(str(candidate),ws,fit=False,params_override=params,with_test=a.include_test,compile_=False)
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status']!='ok':raise RuntimeError(result)
        print(json.dumps(result,indent=2));return
    import torch
    from dftsr.api import load_candidate,params_to_tensors,FAMILIES
    from dftsr import xc_torch as X
    from dftsr.dataset import load_reactions,molecules_for
    from dftsr.features import read_molecule,density_from_arrays
    from scripts.scf_evaluate import summarize
    func=load_candidate(str(candidate));p=params_to_tensors(params,'cuda');fam=FAMILIES[func.family]
    reacts=[r for r in load_reactions() if r.split in splits];recs={}
    logfile=a.out/'molecules.jsonl'
    if logfile.exists():
        for line in logfile.read_text().splitlines():
            r=json.loads(line);recs[r['name']]=r
    with logfile.open('a') as out:
        for name in molecules_for(reacts):
            if name in recs:continue
            kind,d,attrs,_=read_molecule(name);dens=density_from_arrays(kind,d,device='cuda')
            with torch.no_grad():
                w=torch.as_tensor(d['w'],device='cuda')
                ec=float((w*func.energy_density(p,dens)).sum());er=float((w*X.e_xc_b97m_family(dens)).sum())
            hf=fam['cx_sr']*(attrs['E_xx']-attrs['E_xx_lr'])+fam['cx_lr']*attrs['E_xx_lr']
            href=.15*(attrs['E_xx']-attrs['E_xx_lr'])+attrs['E_xx_lr']
            rec=dict(name=name,status='ok',E_nonscf=float(attrs['E_tot']+hf-href+ec-er))
            if not math.isfinite(rec['E_nonscf']):raise RuntimeError(f'Nonfinite molecular energy: {name}')
            recs[name]=rec;out.write(json.dumps(rec)+'\n');out.flush()
    result={sp:summarize([r for r in reacts if r.split==sp],recs,'E_nonscf') for sp in splits}
    result['tot_validation_test_no_RG10']=summarize([r for r in reacts if r.split in ('validation','test') and r.subset!='RG10'],recs,'E_nonscf') if a.include_test else None
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
