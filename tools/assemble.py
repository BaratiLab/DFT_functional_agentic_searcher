"""Assembly record; only writes under this new repository. Not needed by readers."""
from pathlib import Path
import ast, hashlib, json, shutil, sqlite3
R=Path(__file__).resolve().parents[1]
P=R.parent
origins={"opus":Path('/home/ash/Documents/dft_functional_sr'),"codex":Path('/home/ash/Documents/dft_functional_sr_codex')}
manifest=[]
def copy(src,dst):
    dst=R/dst; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
    manifest.append(dict(source=str(src),path=str(dst.relative_to(R)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
for name,src in origins.items():
    for sub in ['dftsr','baselines','mcp_server','prompts','docs']:
        for f in (src/sub).glob('*'):
            if f.suffix in ('.py','.md'): copy(f,Path('harnesses')/name/sub/f.name)
    scripts=['compute_densities.py','compute_constraint_data.py','density_progress.py','build_feature_cache.py','establish_baseline.py','scf_evaluate.py','scf_split_report.py','scf_watch.py']
    scripts+=['gscdb_scf.py','syfes_run.py','syfes_recheck.py','syfes_job.sh'] if name=='opus' else ['run_eval_queue.py','session_mcp.py']
    for f in scripts: copy(src/'scripts'/f,Path('harnesses')/name/'scripts'/f)
    # Archive evaluated sources and fitted coefficients directly from read-only SQLite.
    con=sqlite3.connect(f'file:{src}/runs/iterations.sqlite?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
    ids=[1,2,3,145,159,178,181,182,184,185,186,195,338,376,388,396] if name=='opus' else [1,2,3,339,350,353,356,384,395,398]
    for i in ids:
        row=con.execute('SELECT * FROM iterations WHERE id=?',(i,)).fetchone()
        if row is None: continue
        key=('O' if name=='opus' else 'C')+str(i)
        d=R/'functionals';d.mkdir(exist_ok=True)
        (d/(key+'.py')).write_text(row['code'])
        (d/(key+'.params.json')).write_text(json.dumps(json.loads(row['params_json']),indent=2)+'\n')
        meta={k:row[k] for k in ['id','name','status','train_wrmsd','val_wrmsd','n_violations']}
        meta['source']=f'{name}/runs/iterations.sqlite::iterations[{i}]';meta['harness']=name
        (d/(key+'.metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')
    rows=[dict(r) for r in con.execute('SELECT id,name,status,parent_id,island,train_wrmsd,val_wrmsd,n_violations,tokens_total,eval_seconds FROM iterations')]
    dest=R/'results'/name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'iterations.json').write_text(json.dumps(rows,indent=2)+'\n');con.close()
for key,fn in [('GAS22','gas22'),('wB97M-V','wb97mv'),('B97M-V','b97mv')]:
    f=origins['opus']/'baselines'/(fn+'.py');copy(f,Path('functionals')/(key+'.py'))
    tree=ast.parse(f.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='init_params')
    call=next(n.value for n in node.body if isinstance(n,ast.Return))
    params={k.arg:ast.literal_eval(k.value) for k in call.keywords}
    (R/'functionals'/(key+'.params.json')).write_text(json.dumps(params,indent=2)+'\n')
gold=Path('/home/ash/Documents/dft_fucntional_sr_gold_std')
copy(gold/'runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py','harnesses/opus/scripts/ae11_cpu_fallback.py')
for job in ['job0001_it159_datasets','job0002_it396_datasets','job0003_GAS22_datasets','job0004_GAS22_SCF_datasets']:
    for fn in ['result.json','REPORT.md','run_config.json','params.json','molecules.jsonl']:
        copy(gold/'runs'/job/fn,Path('results/gscdb')/job/fn)
cfg=json.loads((gold/'runs/job0001_it159_datasets/run_config.json').read_text())
(R/'datasets').mkdir(exist_ok=True)
(R/'datasets/gscdb36.txt').write_text(cfg['datasets'].replace(',','\n')+'\n')
for fn in ['mgcdb84_reactions_syfes.csv','mgcdb84_reactions_agentic2026.csv']:
    copy(P/'reproducibility/data'/fn,Path('datasets')/fn)
for fn in ['mgcdb84_absolute_energies.csv','gscdb137_subset_absolute_energies.csv','gscdb_convergence_repairs.csv']:
    copy(P/'reproducibility/data/energies'/fn,Path('results')/fn)
for fn in ['gscdb_it178_summary.json','window_descriptor_analysis.json','window_error_decomposition.txt']:
    copy(origins['opus']/'runs/window_design_rule'/fn,Path('results/window')/fn)
for fn in ['REPORT.md','recheck.json','history.jsonl']:
    copy(origins['opus']/'runs/syfes/job0001_gas22_400'/fn,Path('results/syfes')/fn)
for fn in ['README.md','reevaluation.json','gscdb_summary.json']:
    copy(origins['opus']/'runs/attenuation_fix'/fn,Path('results/attenuation')/fn)
# Historic convergence repair scripts retained as archival evidence, not portable entrypoints.
for f in (P/'reproducibility/source_snapshot/gscdb/runs/convergence_fix').glob('*.py'):
    copy(f,Path('archive/convergence_fix')/f.name)
(R/'provenance').mkdir(exist_ok=True)
for n,f in [('opus',origins['opus']/'project_master_details.md'),('codex',origins['codex']/'project_details.md'),('gscdb',gold/'project_details.md')]:
    copy(f,Path('provenance')/(n+'_project_details.md'))
# Only portability/dispatch changes below; candidate expressions and fitted values are not rewritten.
for n in origins:
    f=R/'harnesses'/n/'dftsr/config.py';s=f.read_text().replace("'/media/ash/PortableSSD/dft_sr'","os.path.join(PROJECT_ROOT, '..', '..', 'work', 'data')")
    f.write_text(s)
f=R/'harnesses/opus/dftsr/gscdb.py';s=f.read_text().replace("'/home/ash/Documents/dft_fucntional_sr_gold_std'","os.path.join(os.path.dirname(__file__), '..', '..', '..', 'work', 'gscdb')").replace("'/media/ash/PortableSSD/dft_sr/gscdb_ref_wb97mv'","os.path.join(GSCDB_WORK, 'reference')");f.write_text(s)
f=R/'harnesses/opus/scripts/ae11_cpu_fallback.py';s=f.read_text().replace("sys.path.insert(0, '/home/ash/Documents/dft_functional_sr')","sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))").replace("RUNS = '/home/ash/Documents/dft_fucntional_sr_gold_std/runs/'","RUNS = G.GSCDB_RUNS + os.sep").replace("default='/home/ash/Documents/dft_functional_sr/baselines/gas22.py'","default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'baselines', 'gas22.py')");f.write_text(s)
f=R/'harnesses/opus/scripts/gscdb_scf.py';s=f.read_text().replace("ap.add_argument('--candidate',", "ap.add_argument('--exclude-molecules', default='', help='comma-separated molecules handled separately on CPU')\n    ap.add_argument('--candidate',");s=s.replace("todo = [m for m in mols if m not in done]","excluded = set(a.exclude_molecules.split(','))\n    todo = [m for m in mols if m not in done and m not in excluded]");f.write_text(s)
f=R/'harnesses/opus/scripts/syfes_job.sh';s=f.read_text().replace('PY=${PYTHON:-/home/ash/Documents/dft_functional_sr/.venv/bin/python}','PY=${PYTHON:-python}').replace('cd /home/ash/Documents/dft_functional_sr','cd "$(dirname "$0")/.."');f.write_text(s)
for fn in ['run_eval_queue.py','session_mcp.py']:
    f=R/'harnesses/codex/scripts'/fn;s=f.read_text().replace("command=str(ROOT/'.venv/bin/python')","command=sys.executable").replace("'DFTSR_SCF_RUNS_DIR':'/media/ash/PortableSSD/dft_sr/scf_runs_codex_run2'","'DFTSR_SCF_RUNS_DIR':os.environ.get('DFTSR_SCF_RUNS_DIR',str(ROOT/'runs/scf'))").replace("'DFTSR_GPUS':'0,1,2'","'DFTSR_GPUS':os.environ.get('DFTSR_GPUS','0')");f.write_text(s)
for item in manifest:
    item['packaged_sha256']=hashlib.sha256((R/item['path']).read_bytes()).hexdigest()
    item['modified_for_portability']=item['packaged_sha256']!=item['source_sha256']
(R/'provenance/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Assembled {len(manifest)} source artifacts under {R}')
