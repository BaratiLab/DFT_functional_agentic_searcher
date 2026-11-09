#!/usr/bin/env python
"""Build GPU feature caches from the per-molecule density files.
   python scripts/build_feature_cache.py --splits train,validation [--allow-partial]
Without --allow-partial the build refuses to run while molecules of the split are still missing."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dftsr.features import build_cache, available
from dftsr.dataset import load_reactions, molecules_for

ap = argparse.ArgumentParser()
ap.add_argument('--splits', default='train,validation')
ap.add_argument('--allow-partial', action='store_true')
a = ap.parse_args()
R = load_reactions()
for sp in a.splits.split(','):
    ms = molecules_for([r for r in R if r.split == sp])
    miss = [m for m in ms if not available(m)]
    if miss and not a.allow_partial:
        print(f'{sp}: {len(miss)}/{len(ms)} molecules missing (e.g. {miss[:3]}); use --allow-partial to build anyway')
        continue
    build_cache(sp)
