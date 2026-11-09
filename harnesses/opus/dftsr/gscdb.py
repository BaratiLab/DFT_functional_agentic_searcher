"""GSCDB137 (Liang & Head-Gordon, JCTC 2025): molecule setup and the published analysis protocol.

Data: https://github.com/JiashuLiang/GSCDB, cloned into $GSCDB_ROOT. The Q-Chem input files are the authority
(the README says every other format is derived from them), so each molecule is built from `qchem_inputs/<name>.in`:
geometry, charge, multiplicity, basis (named or an explicit $basis section), $ecp, $multipole_field, XC grid,
VV10 grid, SCF_GUESS_MIX and INTERNAL_STABILITY. All GSCDB inputs are UNRESTRICTED.

The analysis is a line-by-line port of `Analysis/analyze.ipynb`: reaction energies from `Info/DatasetEval.csv`
(SC74 and OEEFD excluded) -> unit conversion of the finite-difference sets (Dip146 dipoles, Pol130/HR46/T144
polarizabilities, V30 frequencies) -> per-set RMSE/MAE/MSE and the paper's metric (MAE; relative MAE for
Pol130/HR46/T144/OEEF; weighted MAE for TMD10/MOR13/TMB11 and O24/O24x4; regularised relative MAE for Dip146)
-> normalized error ratio NER = metric / standard error (`Info/Standard_errors.csv`) -> mean NER overall and per
data type. The published energies of 16 functionals (`Analysis/Molecule_Energies.csv`, Q-Chem) are loaded for
comparison; their ωB97M-V column also serves as the reference against which our ωB97M-V energies are checked.
"""
from __future__ import annotations

import csv
import functools
import json
import os
import re

import numpy as np

GSCDB_WORK = os.environ.get('GSCDB_WORK', os.path.join(os.path.dirname(__file__), '..', '..', '..', 'work', 'gscdb'))
GSCDB_ROOT = os.environ.get('GSCDB_ROOT', os.path.join(GSCDB_WORK, 'GSCDB'))
GSCDB_RUNS = os.environ.get('GSCDB_RUNS', os.path.join(GSCDB_WORK, 'runs'))
# converged ωB97M-V orbitals, reused as the SCF start for every candidate (≈ 93 GB for all of GSCDB)
GSCDB_REF_DIR = os.environ.get('GSCDB_REF_DIR', os.path.join(GSCDB_WORK, 'reference'))

HARTREE_TO_KCAL = 627.50947406           # value used by analyze.ipynb
EXCLUDED_DATASETS = ('SC74', 'OEEFD')    # dropped by analyze.ipynb
# Q-Chem $multipole_field sign: H = H0 + FIELD_SIGN * F·(Σ_i r_i - Σ_A Z_A R_A) (atomic units, origin at 0).
# Fixed empirically against the published Q-Chem ωB97M-V energies of the Dip146 field pairs (see GSCDB_NOTES.md).
FIELD_SIGN = float(os.environ.get('GSCDB_FIELD_SIGN', '-1'))


# ============================================================================ Q-Chem input parsing
def _section(text, name):
    m = re.search(r'^\s*\$' + name + r'\b(.*?)^\s*\$end', text, re.S | re.I | re.M)
    return m.group(1) if m else None


def _parse_rem(text):
    rem = {}
    for m in re.finditer(r'^\s*\$rem\b(.*?)^\s*\$end', text, re.S | re.I | re.M):
        for line in m.group(1).splitlines():
            line = line.split('!')[0].replace('=', ' ').split()
            if len(line) >= 2:
                rem[line[0].lower()] = line[1]
    return rem


def _parse_grid(s, default=None):
    """'000099000590' -> (99, 590); SG-n codes ('1') -> 'sg1'."""
    if s is None:
        return default
    s = s.strip()
    if len(s) == 12 and s.isdigit():
        return int(s[:6]), int(s[6:])
    if s in ('1', 'sg1', 'SG1'):
        return 'sg1'
    raise ValueError(f'unsupported grid spec {s!r}')


def _split_elements(block):
    """Split a Q-Chem $basis / $ecp body into {element: text} (blocks separated by '****', header 'El 0')."""
    out = {}
    for chunk in block.split('****'):
        lines = [l for l in chunk.strip().splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].split()
        if len(head) > 2 or (len(head) == 2 and head[1] != '0'):
            raise ValueError(f'only element-wide basis/ECP blocks are supported, got header {lines[0]!r}')
        out[head[0].capitalize()] = '\n'.join(lines[1:])
    return out


def _parse_gaussian_ecp(text):
    """Gaussian/Q-Chem ECP text (without the element header) -> PySCF internal ECP format."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = lines[0].split()            # e.g. 'Y-ECP 4 28'
    lmax, ncore = int(head[1]), int(head[2])
    i, blocks = 1, []
    for k in range(lmax + 1):
        i += 1                         # 'g potential' / 's-g potential' title line
        n = int(lines[i].split()[0])
        i += 1
        by_r = [[] for _ in range(7)]
        for j in range(n):
            p, e, c = lines[i + j].replace('D', 'E').split()[:3]
            if float(c) != 0.0:
                by_r[int(p)].append([float(e), float(c)])
        i += n
        blocks.append([-1 if k == 0 else k - 1, by_r])   # first block is the local (lmax) part
    return [ncore, blocks]


def _parse_molecule(text):
    body = [l.strip() for l in _section(text, 'molecule').strip().splitlines() if l.strip()]
    charge, mult = (int(x) for x in body[0].split()[:2])
    atoms = []
    for l in body[1:]:
        p = l.split()
        atoms.append((p[0].capitalize(), float(p[1]), float(p[2]), float(p[3])))
    return atoms, charge, mult


@functools.lru_cache(maxsize=None)
def _info_json():
    return json.load(open(os.path.join(GSCDB_ROOT, 'Allmols_info.json')))


def molecule_spec(name):
    """Everything needed to set up one GSCDB molecule, read from its Q-Chem input."""
    text = open(os.path.join(GSCDB_ROOT, 'qchem_inputs', f'{name}.in')).read()
    atoms, charge, mult = _parse_molecule(text)
    rem = _parse_rem(text)
    field = {}
    mf = _section(text, 'multipole_field')
    if mf:
        for l in mf.strip().splitlines():
            p = l.split()
            if len(p) == 2 and float(p[1]) != 0.0:
                field[p[0].upper()] = float(p[1])
    b = _section(text, 'basis')
    e = _section(text, 'ecp')
    info = _info_json().get(name, {})
    return dict(
        name=name, atoms=atoms, charge=charge, mult=mult,
        basis_name=rem.get('basis', info.get('basis')),
        basis_text=_split_elements(b) if b else None,
        ecp_text=_split_elements(e) if e else None,
        field=field,
        xc_grid=_parse_grid(rem.get('xc_grid'), (99, 590)),
        nl_grid=_parse_grid(rem.get('nl_grid'), 'sg1'),     # Q-Chem default VV10 grid is SG-1
        guess_mix={'true': 1, 'false': 0}.get(rem.get('scf_guess_mix', '0').lower()) if rem.get(
            'scf_guess_mix', '0').lower() in ('true', 'false') else int(rem.get('scf_guess_mix', '0')),
        stability=rem.get('internal_stability', 'false').lower() == 'true',
        scf_convergence=int(rem.get('scf_convergence', '7')),
        num_basis=info.get('num_basis'),
        num_pairs=info.get('num_pairs'),
    )


# ============================================================================ PySCF molecule
_PYSCF_NAMES = {'aug-pc-3', 'aug-pc-4', 'aug-cc-pvqz', 'aug-cc-pv5z', 'aug-cc-pwcvqz', 'def2-qzvppd', 'def2-qzvpp',
                'def2-tzvpd'}     # named GSCDB basis sets shipped with PySCF; the rest come from $basis or BSE


def _named_basis(name, el):
    """One element of a named basis: PySCF library first, then the Basis Set Exchange. Core-valence sets
    (cc-pCVnZ, cc-pwCVnZ) are not defined for H/He; like Q-Chem we fall back to the valence set there."""
    from pyscf import gto
    try:
        return gto.basis.load(name.lower(), el)
    except Exception:
        pass
    import basis_set_exchange as bse
    try:
        txt = bse.get_basis(name, elements=[el], fmt='nwchem', header=False)
    except KeyError:
        if gto.charge(el) > 2 or not re.search(r'cc-p(w)?cv', name, re.I):
            raise
        name = re.sub(r'cc-p(w)?cv', 'cc-pv', name, flags=re.I)
        txt = bse.get_basis(name, elements=[el], fmt='nwchem', header=False)
    return gto.basis.parse(txt, symb=el)


def _block_basis(text, el):
    """A $basis element block: explicit Gaussian-format shells, or a single basis-set name (DAPD inputs)."""
    from pyscf.gto.basis import parse_gaussian
    t = text.strip()
    if len(t.split()) == 1:
        return _named_basis(t, el)
    return parse_gaussian.parse(t)


def build_mol(spec, verbose=0, max_memory=60000):
    from pyscf import gto
    elements = sorted({a[0] for a in spec['atoms']})
    ecp = None
    if spec['basis_text']:
        basis = {el: _block_basis(spec['basis_text'][el], el) for el in elements}
        if spec['ecp_text']:
            ecp = {el: _parse_gaussian_ecp(t) for el, t in spec['ecp_text'].items() if el in elements}
    elif spec['basis_name'].lower() in _PYSCF_NAMES:
        basis = spec['basis_name'].lower()
        if basis.startswith('def2'):       # def2 ECPs for Z > 36
            ecp = {el: basis for el in elements if gto.charge(el) > 36} or None
    else:                                  # e.g. AUG-CC-PCV5Z (not shipped with PySCF)
        basis = {el: _named_basis(spec['basis_name'], el) for el in elements}
    mol = gto.M(atom=[(a[0], a[1:]) for a in spec['atoms']], unit='Angstrom', basis=basis, ecp=ecp,
                charge=spec['charge'], spin=spec['mult'] - 1, cart=False, verbose=verbose, max_memory=max_memory)
    return mol


# ============================================================================ uniform electric field
def apply_field(mf, field, sign=None):
    """Add a uniform field (Q-Chem $multipole_field X/Y/Z, a.u.) to a (GPU4)PySCF mean-field object."""
    if not field:
        return mf
    import cupy as cp
    sign = FIELD_SIGN if sign is None else sign
    F = np.array([field.get('X', 0.0), field.get('Y', 0.0), field.get('Z', 0.0)])
    mol = mf.mol
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor_symmetric('int1e_r', comp=3)
    v = sign * np.einsum('x,xij->ij', F, r)           # electrons: charge -1, +F·r enters with `sign`
    e_nuc_f = -sign * float(np.einsum('a,ax,x->', mol.atom_charges(), mol.atom_coords(), F))
    h0 = mf.get_hcore
    e0 = mf.energy_nuc
    vg = cp.asarray(v)
    mf.get_hcore = lambda *a, **k: h0(*a, **k) + vg
    mf.energy_nuc = lambda *a, **k: e0(*a, **k) + e_nuc_f
    return mf


# ============================================================================ dataset tables
@functools.lru_cache(maxsize=None)
def load_entries():
    """[(reaction, dataset, reference_hartree, [(coef, molecule), ...])] without SC74/OEEFD."""
    out = []
    for r in csv.DictReader(open(os.path.join(GSCDB_ROOT, 'Info', 'DatasetEval.csv'))):
        if r['Dataset'] in EXCLUDED_DATASETS:
            continue
        s = r['Stoichiometry'].split(',')
        out.append((r['Reaction'], r['Dataset'], float(r['Reference']),
                    [(float(s[i]), s[i + 1]) for i in range(0, len(s), 2)]))
    return out


@functools.lru_cache(maxsize=None)
def dataset_table():
    out = {}
    for r in csv.DictReader(open(os.path.join(GSCDB_ROOT, 'Info', 'Datasets.csv'))):
        if r['Name']:
            out[r['Name']] = dict(datatype=r['Datatype'], short=r['Datatype_Short'], n=r['#datapoints'])
    return out


def molecules_for(datasets=None):
    seen, out = set(), []
    for _, ds, _, st in load_entries():
        if datasets and ds not in datasets:
            continue
        for _, m in st:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def select_molecules(subset='all', datasets=None, max_nbas=0):
    """subset: 'all' | 'probe' (validation set of ~40 molecules covering every special case) | 'datasets'
    (only the comma-separated `datasets`). Returned smallest-first by Q-Chem basis-function count."""
    info = _info_json()
    if subset == 'probe':
        mols = probe_molecules()
    else:
        ds = None
        if datasets:
            ds = {d.strip() for d in (datasets.split(',') if isinstance(datasets, str) else datasets) if d.strip()}
            unknown = ds - {e[1] for e in load_entries()}
            if unknown:
                raise ValueError(f'unknown datasets: {sorted(unknown)}')
        mols = molecules_for(ds)
    if max_nbas:
        mols = [m for m in mols if info[m]['num_basis'] <= max_nbas]
    return sorted(mols, key=lambda m: info[m]['num_basis'])


def probe_molecules():
    """Validation probe: the 1–2 smallest molecules for every special case (each named basis, explicit
    $basis/$ecp, def2 ECPs, each XC/VV10 grid, fields, guess mixing, stability flag, high spin, transition
    metals). Deterministic; about 40 molecules."""
    info = _info_json()
    by = {}
    names = sorted(molecules_for(), key=lambda m: info[m]['num_basis'])
    for m in names:
        s = molecule_spec(m)
        keys = [f'basis:{s["basis_name"]}', f'grid:{s["xc_grid"]}', f'nl:{s["nl_grid"]}']
        if s['ecp_text'] or (s['basis_name'] or '').startswith('def2') and any(
                a[0] in _HEAVY for a in s['atoms']):
            keys.append('ecp')
        if s['field']:
            keys.append('field')
        if s['guess_mix']:
            keys.append(f'guess_mix:{"closed" if s["mult"] == 1 else "open"}')
        if s['stability']:
            keys.append(f'stability:{"closed" if s["mult"] == 1 else "open"}')
        if s['mult'] > 2:
            keys.append(f'mult>{2}')
        if any(a[0] in _TM for a in s['atoms']):
            keys.append('tm')
        for k in keys:
            if len(by.setdefault(k, [])) < 2:
                by[k].append(m)
    out = []
    for v in by.values():
        for m in v:
            if m not in out:
                out.append(m)
    return out


_TM = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
       'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
_HEAVY = {'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
          'Cs', 'Ba', 'La', 'Yb', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Rn'}


@functools.lru_cache(maxsize=None)
def published_energies():
    """{functional: {molecule: E_h}} from Analysis/Molecule_Energies.csv (Q-Chem, GSCDB protocol)."""
    path = os.path.join(GSCDB_ROOT, 'Analysis', 'Molecule_Energies.csv')
    rows = list(csv.reader(open(path)))
    funcs = rows[0][1:]
    out = {f: {} for f in funcs}
    for r in rows[1:]:
        for f, v in zip(funcs, r[1:]):
            if v not in ('', 'nan', 'NaN'):
                out[f][r[0]] = float(v)
    return out


# ============================================================================ analysis (port of analyze.ipynb)
_AU_TO_DEBYE = 2.541746
_AU_TO_A3 = 0.529177 ** 3


def _to_dipole(e_kcal, step=1e-4):
    return e_kcal / HARTREE_TO_KCAL / (2 * step) * _AU_TO_DEBYE


def _to_polarizability(e_kcal, step):
    return e_kcal / HARTREE_TO_KCAL / step ** 2 * _AU_TO_A3


def _to_frequency(e_kcal, step):
    fc = e_kcal / HARTREE_TO_KCAL
    sgn = np.sign(fc)
    fc = abs(fc) / step ** 2 / (1.66053907e-27 / 9.10938370e-31 * (1 / 0.529177211) ** 2)
    return sgn * np.sqrt(fc) / 4.55633528e-6


@functools.lru_cache(maxsize=None)
def _side_tables():
    info = os.path.join(GSCDB_ROOT, 'Info')
    fd = {r['Index']: float(r['fd step']) for r in csv.DictReader(open(os.path.join(info, 'FD_stepsize.csv')))}
    w = {}
    for f in ('DatasetEval_TMC34_weight.csv', 'DatasetEval_O24x5_weight.csv'):
        for r in csv.DictReader(open(os.path.join(info, f))):
            w.setdefault(r['Dataset'], []).append(float(r['Weight']))
    std = {r['Dataset']: float(r['Metric']) for r in csv.DictReader(open(os.path.join(info, 'Standard_errors.csv')))}
    return fd, w, std


def property_values(energies):
    """{reaction: (dataset, reference_value, computed_value)} in the notebook's common units, for every entry
    whose molecules all have energies (energies: {molecule: E_h})."""
    fd, _, _ = _side_tables()
    out = {}
    for rxn, ds, ref, st in load_entries():
        if not all(m in energies for _, m in st):
            continue
        val = sum(c * energies[m] for c, m in st) * HARTREE_TO_KCAL
        ref = ref * HARTREE_TO_KCAL
        if ds == 'Dip146':
            val, ref = _to_dipole(val), _to_dipole(ref)
        elif ds == 'Pol130':
            val, ref = _to_polarizability(val, fd[rxn]), _to_polarizability(ref, fd[rxn])
        elif ds in ('HR46', 'T144'):
            val, ref = _to_polarizability(val, 0.004), _to_polarizability(ref, 0.004)
        elif ds == 'V30':
            val, ref = _to_frequency(val, fd[rxn]), _to_frequency(ref, fd[rxn])
        out[rxn] = (ds, ref, val)
    return out


def set_metrics(energies):
    """Per-dataset RMSE/MAE/MSE/metric/NER over the datasets that are complete for this energy table."""
    _, w, std = _side_tables()
    n_per_set = {}
    for _, ds, _, _ in load_entries():
        n_per_set[ds] = n_per_set.get(ds, 0) + 1
    vals = property_values(energies)
    by = {}
    for rxn, (ds, ref, val) in vals.items():
        by.setdefault(ds, []).append((ref, val))
    dt = dataset_table()
    out = {}
    for ds, rows in by.items():
        if len(rows) != n_per_set[ds]:
            continue                                   # incomplete set: excluded, as in the notebook
        ref = np.array([r for r, _ in rows])
        err = np.array([v for _, v in rows]) - ref
        mae = float(np.mean(np.abs(err)))
        metric = mae
        if ds in ('Pol130', 'HR46', 'T144', 'OEEF'):
            metric = float(np.mean(np.abs(err / ref)))
        elif ds in ('TMD10', 'MOR13', 'TMB11'):
            wt = np.array(w[ds])
            metric = float(np.sum(np.abs(err) * wt[:-1]) + wt[-1])
        elif ds in ('O24', 'O24x4'):
            metric = float(np.sum(np.abs(err) * np.array(w[ds])))
        elif ds == 'Dip146':
            metric = float(np.mean(np.abs(err / np.maximum(np.abs(ref), 1))))
        out[ds] = dict(datatype=dt.get(ds, {}).get('datatype', ''), n=len(rows), rmse=float(np.sqrt(np.mean(err ** 2))),
                       mae=mae, mse=float(np.mean(err)), metric=metric, ner=metric / std[ds])
    return out


def summarize(sets, restrict_to=None):
    """Mean NER overall and per data type (optionally over a fixed list of datasets, for fair comparisons)."""
    keys = [k for k in sets if restrict_to is None or k in restrict_to]
    if not keys:
        return dict(n_sets=0)
    out = dict(n_sets=len(keys), mean_ner=float(np.mean([sets[k]['ner'] for k in keys])))
    by = {}
    for k in keys:
        by.setdefault(sets[k]['datatype'], []).append(sets[k]['ner'])
    out['mean_ner_by_type'] = {t: dict(n_sets=len(v), mean_ner=float(np.mean(v))) for t, v in sorted(by.items())}
    return out


def compare(tables, include_published=True):
    """tables: {method: {molecule: E_h}}. Returns per-method set metrics plus a ranking table restricted to the
    datasets complete for EVERY method in `tables` (published functionals are complete everywhere)."""
    methods = dict(tables)
    if include_published:
        for f, e in published_energies().items():
            methods.setdefault(f'{f} (published)', e)
    sets = {m: set_metrics(e) for m, e in methods.items()}
    common = set.intersection(*(set(s) for m, s in sets.items() if m in tables)) if tables else None
    ranking = sorted(((m, summarize(s, common)) for m, s in sets.items()), key=lambda x: x[1].get('mean_ner', 1e9))
    return dict(common_datasets=sorted(common) if common is not None else None,
                ranking=[dict(method=m, **v) for m, v in ranking], sets=sets)
