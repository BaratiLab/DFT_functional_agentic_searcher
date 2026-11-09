#!/usr/bin/env python3
"""Reaggregate archived GSCDB energies with the pinned external benchmark inputs (CPU)."""
import json
from runtime import ROOT,configure
configure()
from dftsr import gscdb as G
from dftsr.gscdb_jobs import read_records,finalize
names=(ROOT/'datasets/gscdb36.txt').read_text().split()
mols=G.select_molecules('datasets',names)
assert len(names)==36 and len(mols)==3487
for job in sorted((ROOT/'results/gscdb').iterdir()):
    old=json.loads((job/'result.json').read_text())
    cfg=json.loads((job/'run_config.json').read_text())
    new=finalize(read_records(str(job/'molecules.jsonl')),mols,cfg['label'])
    assert new['n_ok']==3487 and len(new['common_datasets'])==36,job.name
    # Compare per-set data to avoid relying on report formatting or rounding.
    delta=0.
    for method,sets in new['sets'].items():
        if method not in old['sets']:continue
        for name in names:
            delta=max(delta,abs(sets[name]['ner']-old['sets'][method][name]['ner']))
    assert delta<1e-9,(job.name,delta)
    print(job.name,'3487 molecules, 36 sets; maximum NER difference',delta)
