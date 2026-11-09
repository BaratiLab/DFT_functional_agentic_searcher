"""MGCDB84 dataset: reactions, stoichiometry, train/validation/test splits and WRMSD weights.

Splits, data types and weights follow SyFES (Ma et al., Sci. Adv. 2022), which in turn follow
wB97M-V (Mardirossian & Head-Gordon, JCP 144, 214110 (2016)):
  training = "training set", validation = "primary test set", test = "secondary test set".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

from .config import ACCDB_DIR, MGCDB84_DIR, DATA_ROOT

HARTREE_TO_KCAL = 627.509474

MGCDB84_INFO = {
    'NCED': {'train': ['A24', 'DS14', 'HB15', 'HSG', 'NBC10', 'S22', 'X40'],
             'validation': ['A21x12', 'BzDC215', 'HW30', 'NC15', 'S66', 'S66x8'],
             'test': ['3B-69-DIM', 'AlkBind12', 'CO2Nitrogen16', 'HB49', 'Ionic43']},
    'NCEC': {'train': ['H2O6Bind8', 'HW6Cl', 'HW6F'],
             'validation': ['FmH2O10', 'Shields38', 'SW49Bind345', 'SW49Bind6', 'WATER27'],
             'test': ['3B-69-TRIM', 'CE20', 'H2O20Bind10', 'H2O20Bind4']},
    'NCD': {'train': ['TA13', 'XB18'], 'validation': ['Bauza30', 'CT20', 'XB51'], 'test': []},
    'IE': {'train': ['AlkIsomer11', 'Butanediol65'],
           'validation': ['ACONF', 'CYCONF', 'Pentane14', 'SW49Rel345', 'SW49Rel6'],
           'test': ['H2O16Rel5', 'H2O20Rel10', 'H2O20Rel4', 'Melatonin52', 'YMPJ519']},
    'ID': {'train': ['EIE22', 'Styrene45'], 'validation': ['DIE60', 'ISOMERIZATION20'], 'test': ['C20C24']},
    'TCE': {'train': ['AlkAtom19', 'BDE99nonMR', 'G21EA', 'G21IP', 'TAE140nonMR'],
            'validation': ['AlkIsod14', 'BH76RC', 'EA13', 'HAT707nonMR', 'IP13', 'NBPRC', 'SN13'],
            'test': ['BSR36', 'HNBrBDE18', 'WCPT6']},
    'TCD': {'train': [], 'validation': ['BDE99MR', 'HAT707MR', 'TAE140MR'],
            'test': ['PlatonicHD6', 'PlatonicID6', 'PlatonicIG6', 'PlatonicTAE6']},
    'BH': {'train': ['BHPERI26', 'CRBH20', 'DBH24'], 'validation': ['CR20', 'HTBH38', 'NHTBH38'],
           'test': ['PX13', 'WCPT27']},
    'AE18': {'train': ['AE18'], 'validation': [], 'test': []},
    'RG10': {'train': [], 'validation': ['RG10'], 'test': []},
}

MGCDB84_GROUP_WEIGHTS = {
    ('TCD',): 0.1, ('TCE', 'AE18'): 1, ('NCD',): 10, ('ID',): 10, ('BH',): 10,
    ('NCED', 'NCEC'): 100, ('IE',): 100, ('RG10',): 10000,
}


@dataclass
class Reaction:
    name: str
    subset: str            # e.g. 'S66', 'TAE140nonMR'
    dtype: str             # MGCDB84 data type: NCED, NCEC, NCD, IE, ID, TCE, TCD, BH, AE18, RG10
    split: str             # train | validation | test
    ref: float             # reference energy, Hartree
    weight: float          # WRMSD weight
    stoich: list           # [(coefficient, molecule_name), ...]


def resolve_geometry(mol_name: str) -> str:
    """Map an MGCDB84 molecule name to the ACCDB geometry file name (some characters are escaped)."""
    gdir = os.path.join(ACCDB_DIR, 'Geometries')
    for c in (mol_name, mol_name.replace('=', '_'), mol_name.replace('.', '_'),
              mol_name.replace('.', '_').replace('=', '_')):
        if os.path.exists(os.path.join(gdir, c + '.xyz')):
            return os.path.join(gdir, c + '.xyz')
    raise FileNotFoundError(mol_name)


def read_xyz(path):
    lines = open(path).read().strip().split('\n')
    n = int(lines[0])
    charge, mult = map(int, lines[1].split()[:2])
    return '\n'.join(l.strip() for l in lines[2:2 + n]), charge, mult


def _parse_dataset_eval(path):
    rows = {}
    for line in open(path):
        tok = line.strip().split(',')
        if len(tok) < 3:
            continue
        nsp = len(tok) // 2 - 1
        rows[tok[0]] = dict(subset=tok[0].partition('_')[0],
                            stoich=[(int(tok[2 * i + 1]), tok[2 * i + 2]) for i in range(nsp)],
                            ref=float(tok[-1]))
    return rows


@lru_cache(maxsize=4)
def load_reactions(split_variant: str = 'syfes') -> list[Reaction]:
    """Load all MGCDB84 reactions with SyFES splits & weights.

    split_variant: 'syfes'        -> exact SyFES/wB97M-V split (train 870 / val ~2960 / test ~1150).
                   'agentic2026'  -> as 'syfes' but C20C24 moved test -> validation (Duston et al. 2026).
    """
    rows = _parse_dataset_eval(os.path.join(MGCDB84_DIR, 'DatasetEval.csv'))
    rows.pop('XB51_15', None)  # geometry files missing in the original SI (SyFES drops it too)

    # TAE140 / BDE99 / HAT707 -> MR / nonMR, as in SyFES
    mr_idx = [115, 121] + list(range(127, 141))
    mr_mols = {rows[f'TAE140_{i}']['stoich'][0][1] for i in mr_idx}
    for k, r in rows.items():
        if r['subset'] in ('TAE140', 'BDE99', 'HAT707'):
            is_mr = any(m in mr_mols for _, m in r['stoich'])
            r['subset'] += 'MR' if is_mr else 'nonMR'

    info = {k: {s: list(v) for s, v in d.items()} for k, d in MGCDB84_INFO.items()}
    if split_variant == 'agentic2026':
        info['ID']['test'].remove('C20C24')
        info['ID']['validation'].append('C20C24')
    sub2 = {}
    for dtype, d in info.items():
        for split, names in d.items():
            for n in names:
                sub2[n] = (dtype, split)

    # weights (SyFES port of the wB97M-V scheme)
    by_subset = {}
    for k, r in rows.items():
        by_subset.setdefault(r['subset'], []).append(r['ref'])
    w_sub = {s: 1.0 / (len(v) * np.sqrt(np.mean(np.square(v)))) for s, v in by_subset.items()}
    for group, gw in MGCDB84_GROUP_WEIGHTS.items():
        names = [n for dt in group for v in MGCDB84_INFO[dt].values() for n in v]
        ws = np.array([w_sub[n] for n in names])
        ws = ws / ws.min()
        expo = 1.0 if group == ('RG10',) else np.log(2) / np.log(ws.max())
        for n, wv in zip(names, ws ** expo * gw):
            w_sub[n] = wv

    out = []
    for k, r in rows.items():
        dtype, split = sub2[r['subset']]
        w = w_sub[r['subset']]
        if dtype == 'RG10' and r['ref'] > 0:
            w /= MGCDB84_GROUP_WEIGHTS[('RG10',)]
        if split == 'validation':
            w *= 2
        out.append(Reaction(k, r['subset'], dtype, split, r['ref'], float(w), r['stoich']))
    return out


def molecules_for(reactions) -> list[str]:
    s = []
    seen = set()
    for r in reactions:
        for _, m in r.stoich:
            if m not in seen:
                seen.add(m)
                s.append(m)
    return s


def all_molecules() -> list[str]:
    return open(os.path.join(MGCDB84_DIR, 'MGCDB84_ALL.list')).read().split()


def wrmsd(pred_kcal, ref_kcal, weights):
    pred_kcal, ref_kcal, weights = map(np.asarray, (pred_kcal, ref_kcal, weights))
    return float(np.sqrt(np.mean(weights * (pred_kcal - ref_kcal) ** 2)))


if __name__ == '__main__':
    R = load_reactions()
    from collections import Counter
    print('reactions', len(R), Counter(r.split for r in R))
    for sp in ('train', 'validation', 'test'):
        print(sp, 'molecules', len(molecules_for([r for r in R if r.split == sp])))
    print(Counter((r.dtype, r.split) for r in R))
