#!/usr/bin/env python3
"""CPU checks; no molecular calculation, fitting, or changes to archived results."""
import ast,csv,hashlib,json,math
from pathlib import Path
from collections import Counter
from runtime import ROOT
for f in ROOT.rglob('*.py'):
    if '.venv' not in f.parts:ast.parse(f.read_text(),filename=str(f))
for m in json.loads((ROOT/'provenance/source_manifest.json').read_text()):
    assert hashlib.sha256((ROOT/m['path']).read_bytes()).hexdigest()==m['packaged_sha256'],m['path']
for f in (ROOT/'functionals').glob('*.params.json'):
    p=json.loads(f.read_text());assert p and all(math.isfinite(x) for v in p.values() for x in v)
reactions=list(csv.DictReader((ROOT/'datasets/mgcdb84_reactions_syfes.csv').open()))
assert Counter(r['split'] for r in reactions)=={'train':870,'validation':2963,'test':1152}
energies={r['molecule']:r for r in csv.DictReader((ROOT/'results/mgcdb84_absolute_energies.csv').open())}
summary={}
for method in ['wb97mv_reference_hartree','it159_scf_hartree','it159_nonSCF_hartree','GAS22_scf_hartree','GAS22_nonSCF_hartree']:
    summary[method]={}
    for split in ['train','validation','test']:
        terms=[]
        for r in reactions:
            if r['split']!=split:continue
            e=sum(c*float(energies[m][method]) for c,m in json.loads(r['stoichiometry']))
            terms.append(float(r['weight'])*((e-float(r['reference_hartree']))*627.509474)**2)
        summary[method][split]=dict(n=len(terms),wrmsd=math.sqrt(sum(terms)/len(terms)))
expected={'it159_scf_hartree':(3.330,3.805),'GAS22_scf_hartree':(3.470,3.589),'wb97mv_reference_hartree':(3.533,4.309)}
for method,(val,test) in expected.items():
    assert abs(summary[method]['validation']['wrmsd']-val)<.001,(method,summary[method])
    assert abs(summary[method]['test']['wrmsd']-test)<.001,(method,summary[method])
print(json.dumps(summary,indent=2));print('PASS: Python syntax, source hashes, frozen parameter finiteness, split counts, and saved-energy MGCDB table.')
