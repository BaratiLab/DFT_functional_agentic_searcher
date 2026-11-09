"""Self-consistent ωB97M-V reference and candidate calculations for GSCDB137 molecules (GPU4PySCF).

Protocol per molecule (mirrors the MGCDB84 SCF evaluation, `scripts/scf_evaluate.py`, as closely as GSCDB allows):
  1. ωB97M-V (libxc) SCF in the GSCDB setup: basis/ECP/field/XC grid/VV10 grid from the Q-Chem input, unpruned
     XC grid (Q-Chem convention), RI-JK (def2-universal-jkfit) during the SCF, then the exact-integral energy at
     the converged density, for def2 basis sets up to DF_MAX_NAO functions; everything else uses exact JK.
     Guess: PySCF MINAO; SCF_GUESS_MIX molecules get a broken-symmetry UKS guess (HOMO/LUMO mixing);
     INTERNAL_STABILITY singlets are run both restricted and broken-symmetry, the lower energy is kept.
     Converged orbitals are stored (GSCDB_REF_DIR) and reused by every later candidate.
     The result is checked against the published Q-Chem ωB97M-V energy of the same molecule.
  2. Candidate non-SCF energy at the ωB97M-V density (exact integrals).
  3. Candidate SCF started from the ωB97M-V orbitals (same restricted/unrestricted form), exact-integral energy.

The candidate's semilocal XC goes through `dftsr.scf.TorchNumInt`; exact exchange and VV10 come from the frame
functional ('wb97m-v' for the 'rsh' family), exactly as in the MGCDB84 SCF runs.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import gscdb as G

CONV_TOL = 1e-10
CONV_TOL_TIGHT = 1e-11            # V30 (finite-difference frequencies need ~1e-9 Eh in energy differences)
DF_MAX_NAO = int(os.environ.get('GSCDB_DF_MAX_NAO', '6000'))
# GPU4PySCF keeps the RI 3-index tensor on the GPU when it fits, otherwise in pinned host memory. Estimated size
# 2.2 * nao * n_pairs * 8 B (calibrated: C60ISO_1, nao 2220, 1.98M significant pairs -> 72 GB host, measured).
GPU_DF_GB = float(os.environ.get('GSCDB_GPU_DF_GB', '25'))         # what fits on a 48 GB GPU next to the rest
HOST_BUDGET_GB = float(os.environ.get('GSCDB_HOST_BUDGET_GB', '75'))
# ^ pinned host memory for all workers together. systemd-oomd kills the unit when the user slice's memory
# pressure exceeds 50 % (page cache in the slice must be reclaimed): 2 x C60ISO (≈ 2 x 72 GB) at once was killed
# twice on 2026-10-20, one C60ISO alone ran with zero pressure. User units cannot opt out (ManagedOOMPreference
# is ignored for cgroups not owned by root).


def _set_grids(mf, spec):
    from pyscf.dft import gen_grid
    mf.grids.atom_grid = spec['xc_grid']
    mf.grids.prune = None                         # Q-Chem XC_GRID 000099000590 = unpruned (99, 590)
    if spec['nl_grid'] == 'sg1':                  # SG-1 = pruned (50, 194); PySCF's SG-1 pruning stops at Ar
        from pyscf import gto
        mf.nlcgrids.atom_grid = (50, 194)
        # true nuclear charges: atom_charges() is ECP-reduced (Y-Pd with def2 ECPs: 11-18), but sg1_prune indexes
        # its H-Ar radius table by Z (IndexError for the 4d atoms, 2026-10-21)
        zmax = max(gto.charge(mf.mol.atom_pure_symbol(i)) for i in range(mf.mol.natm))
        mf.nlcgrids.prune = gen_grid.sg1_prune if zmax <= 18 else None
    else:
        mf.nlcgrids.atom_grid = spec['nl_grid']
        mf.nlcgrids.prune = None
    return mf


def _cpu_hcore(mf):
    """GPU4PySCF's one-electron kernels are wrong for shells with l >= 5 (h functions): the ECP kernel fails
    (aug-cc-pwCVQZ-PP on I/Sn), and the point-charge nuclear attraction (1.8.1, contract_int3c2e_auxvec) is off by
    up to ~10 Eh on h elements (aug-pc-4 diatomics of IP23/G21EA collapsed 50-170 Eh below Q-Chem, 2026-10-21).
    For every molecule with h functions, T + V_nuc (+ V_ECP) is built once on the CPU with PySCF and cached."""
    import cupy as cp
    from pyscf.scf import hf as cpu_hf
    mol = mf.mol
    if max(mol.bas_angular(i) for i in range(mol.nbas)) >= 5:
        if getattr(mol, '_gscdb_hcore', None) is None:
            mol._gscdb_hcore = cp.asarray(cpu_hf.get_hcore(mol))
        h = mol._gscdb_hcore
        mf.get_hcore = lambda *a, **k: h
    return mf


def _split_rsh_k(mf):
    """GPU4PySCF 1.8.1's fused range-separated K (get_k with lr_factor/sr_factor, used by get_veff) is wrong for
    shells with l >= 5 (at a fixed density: He-Ar at 1.85 Å 2.7 kcal/mol, IP23 LiF 10 kcal/mol), while the separate
    full-range and long-range K agree with PySCF to 1e-7 kcal/mol (2026-10-21). For molecules with h functions the
    fused call is replaced by lr * K_LR + sr * (K_full - K_LR)."""
    mol = mf.mol
    if max(mol.bas_angular(i) for i in range(mol.nbas)) < 5:
        return mf
    get_k0 = mf.get_k

    def get_k(mol=None, dm=None, hermi=1, omega=None, lr_factor=None, sr_factor=None):
        if lr_factor is None and sr_factor is None:
            return get_k0(mol, dm, hermi, omega)
        k_lr = get_k0(mol, dm, hermi, omega)
        k_full = get_k0(mol, dm, hermi)
        return (lr_factor or 0.0) * k_lr + (sr_factor or 0.0) * (k_full - k_lr)
    mf.get_k = get_k
    return mf


def make_mf(mol, spec, unrestricted, func=None, params=None, density_fit=True):
    """KS object in the GSCDB setup. func=None: libxc ωB97M-V; otherwise a dftsr candidate."""
    from gpu4pyscf import dft
    from .scf import FRAME_XC, TorchNumInt
    K = dft.UKS if unrestricted else dft.RKS
    mf = K(mol, xc='wb97m-v' if func is None else FRAME_XC[func.family])
    if density_fit:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    if func is not None:
        mf._numint = TorchNumInt(func, params)
    _set_grids(mf, spec)
    _cpu_hcore(mf)
    _split_rsh_k(mf)
    G.apply_field(mf, spec['field'])
    mf.conv_tol = CONV_TOL_TIGHT if spec['name'].startswith('V30') else CONV_TOL
    mf.conv_tol_grad = 10.0 ** -(spec['scf_convergence'] - 1)
    mf.max_cycle = 150
    return mf


def _mix_guess(mf, mix):
    """Broken-symmetry UKS start: diagonalise the Fock matrix of the MINAO guess, then rotate the α HOMO towards
    the LUMO and the β HOMO the other way by Q-Chem's SCF_GUESS_MIX fraction (n × 10 %). The generalized
    eigenproblem is solved on the CPU (cuSOLVER's sygvd fails on some of these matrices)."""
    import cupy as cp
    import scipy.linalg
    dm0 = mf.get_init_guess()
    f = mf.get_fock(dm=dm0).get()
    s = mf.get_ovlp().get()
    es, cs = zip(*(scipy.linalg.eigh(f[k], s) for k in range(2)))
    e, c = cp.asarray(np.array(es)), np.array(cs)
    occ = mf.get_occ(e, cp.asarray(c))
    x = mix / 10.0
    for spin, sgn in ((0, 1.0), (1, -1.0)):
        h = int(cp.count_nonzero(occ[spin])) - 1
        a, b = c[spin][:, h].copy(), c[spin][:, h + 1].copy()
        c[spin][:, h] = (a + sgn * x * b) / np.sqrt(1 + x * x)
        c[spin][:, h + 1] = (b - sgn * x * a) / np.sqrt(1 + x * x)
    return mf.make_rdm1(cp.asarray(c), occ)


def _kernel(mf, dm0=None):
    """SCF with fallbacks: plain DIIS -> level shift + damping -> second-order (Newton) solver."""
    mf.kernel(dm0=dm0)
    how = 'diis'
    if not mf.converged:
        mf.level_shift, mf.damp, mf.max_cycle = 0.25, 0.5, 300
        mf.kernel(dm0=mf.make_rdm1())
        how = 'level_shift'
        mf.level_shift, mf.damp = 0.0, 0.0
    if not mf.converged:
        try:
            mf2 = mf.newton()
            mf2.max_cycle = 100
            mf2.kernel(mf.mo_coeff, mf.mo_occ)
            if mf2.converged:
                mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot = mf2.mo_coeff, mf2.mo_occ, mf2.mo_energy, mf2.e_tot
                mf.converged = True
            how = 'newton'
        except Exception as e:  # noqa: BLE001 (record and keep the level-shift result)
            how = f'newton_failed:{type(e).__name__}'
    return how


def exact_energy(mol, spec, unrestricted, dm, func=None, params=None, grids_from=None):
    mfx = make_mf(mol, spec, unrestricted, func, params, density_fit=False)
    if grids_from is not None:
        mfx.grids, mfx.nlcgrids = grids_from.grids, grids_from.nlcgrids
    return float(mfx.energy_tot(dm=dm))


def df_host_gb(mol, spec):
    """Estimated pinned host memory of the RI tensor (0 when it fits on the GPU)."""
    pairs = spec.get('num_pairs') or 0.8 * mol.nao ** 2 / 2
    return max(0.0, 2.2 * mol.nao * pairs * 8 / 1e9 - GPU_DF_GB)


def use_df(mol, spec):
    """RI-JK only where it was validated: def2 basis sets (def2-universal-jkfit is their fitting set; the exact
    energy at the DF density then matches Q-Chem to ~1e-7 Eh). For aug-pc-n, (d-)aug-cc-p(w)CVnZ and explicit
    $basis sets the DF density is off by up to 1e-3 Eh, so those (small) molecules use exact JK throughout.
    Molecules whose RI tensor would not fit the host-memory budget also use exact JK (≈ 3.7x slower at 1344
    functions, and growing: avoid them with max_nbas unless the time is acceptable)."""
    return ((spec['basis_name'] or '').lower().startswith('def2') and mol.nao <= DF_MAX_NAO
            and df_host_gb(mol, spec) <= HOST_BUDGET_GB)


def _scf(mol, spec, unrestricted, func=None, params=None, dm0=None, mix=0):
    df = use_df(mol, spec)
    mf = make_mf(mol, spec, unrestricted, func, params, density_fit=df)
    if dm0 is None and mix:
        dm0 = _mix_guess(mf, mix)
    t = time.time()
    how = _kernel(mf, dm0)
    rec = dict(converged=bool(mf.converged), how=how, cycles=getattr(mf, 'cycles', None),
               e_scf=float(mf.e_tot), df=df, t_scf=round(time.time() - t, 1))
    dm = mf.make_rdm1()
    t = time.time()
    rec['e'] = exact_energy(mol, spec, unrestricted, dm, func, params, mf) if df else float(mf.e_tot)
    rec['t_exact'] = round(time.time() - t, 1)
    if unrestricted:
        try:
            rec['s2'] = float(mf.spin_square()[0])
        except Exception:  # noqa: BLE001
            pass
    return mf, rec


def _follow_instability(mf, rounds=4, log=None):
    """Internal stability analysis on the CPU (GPU4PySCF has none for UKS meta-GGA/VV10; PySCF's response omits
    the VV10 kernel, which is fine for finding the unstable direction), re-SCF on the GPU along it, repeat."""
    import cupy as cp
    for k in range(rounds):
        mc = mf.to_cpu()
        if 'get_hcore' in mc.__dict__:    # _cpu_hcore / apply_field patches return CuPy arrays; to_cpu copies them
            h_gpu = mc.get_hcore
            mc.get_hcore = lambda *a, _h=h_gpu, **kw: cp.asnumpy(_h(*a, **kw))
        mc.__dict__.pop('get_k', None)    # _split_rsh_k (GPU only): PySCF's own get_k is correct
        mo, _, stable, _ = mc.stability(return_status=True)
        if log is not None:
            log.append(dict(round=k, stable=bool(stable), e=float(mf.e_tot)))
        if stable:
            return mf
        e0 = float(mf.e_tot)
        _kernel(mf, cp.asarray(mc.make_rdm1(mo, mc.mo_occ)))   # same fallbacks as every other SCF (3d4dIPSS_Sc_GS+
                                                              # reached +0.056 kcal/mol here but plain DIIS stalled)
        if float(mf.e_tot) > e0 - 1e-6:   # no further lowering (the missing VV10 response can leave a
            break                         # spurious "unstable" direction; Sc+ plateaued at +1.2e-4 Eh this way)
    return mf


def _field_siblings(name, ref_dir=None):
    """Molecules of the same data set with identical atoms, charge, multiplicity, basis and ECP but another field
    (finite-difference property sets), whose ωB97M-V orbitals are stored."""
    s = G.molecule_spec(name)
    key = lambda t: (t['atoms'], t['charge'], t['mult'], t['basis_name'], t['basis_text'], t['ecp_text'])  # noqa: E731
    prefix = name.split('_')[0] + '_'
    out = []
    for n in G._info_json():
        if n != name and n.startswith(prefix) and os.path.exists(ref_path(n, ref_dir)):
            t = G.molecule_spec(n)
            if t['field'] != s['field'] and key(t) == key(s):
                out.append(n)
    return sorted(out)


def _state_search(mol, spec, mf, rec, e_target):
    """Our ωB97M-V landed above the published Q-Chem ωB97M-V energy of the same molecule (e_target): look for the
    lower state (other guesses, then stability following, UKS) and keep the lowest exact energy. Only the
    reference functional's published energy is used here, never benchmark reference data."""
    trials = [dict(guess=rec['guess'], e=rec['e'], converged=rec['converged'])]
    best_mf, best = mf, rec
    for guess in ('huckel', 'atom', 'mix3'):
        if guess == 'mix3' and spec['mult'] != 1:
            continue
        try:
            m = make_mf(mol, spec, True, density_fit=use_df(mol, spec))
            dm0 = _mix_guess(m, 3) if guess == 'mix3' else None
            if guess != 'mix3':
                m.init_guess = guess
            _kernel(m, dm0)
            e = exact_energy(mol, spec, True, m.make_rdm1(), grids_from=m) if use_df(mol, spec) else float(m.e_tot)
            trials.append(dict(guess=guess, e=e, converged=bool(m.converged)))
            if m.converged and e < best['e'] - 1e-7:
                best_mf, best = m, dict(rec, e=e, e_scf=float(m.e_tot), guess=guess, converged=True)
        except Exception as ex:  # noqa: BLE001
            trials.append(dict(guess=guess, error=repr(ex)[:200]))
        if best['e'] - e_target < 1e-5:
            break
    if best['e'] - e_target > 1e-5:
        try:
            if best_mf.istype('UKS'):
                m = best_mf
            else:                       # restricted solution: continue unrestricted from the same density
                import cupy as cp
                m = make_mf(mol, spec, True, density_fit=use_df(mol, spec))
                dm = best_mf.make_rdm1()
                _kernel(m, cp.stack([dm / 2, dm / 2]))
            slog = []
            m = _follow_instability(m, log=slog)
            e = exact_energy(mol, spec, True, m.make_rdm1(), grids_from=m) if use_df(mol, spec) else float(m.e_tot)
            trials.append(dict(guess=f'{best["guess"]}+stability', e=e, converged=bool(m.converged), log=slog))
            if m.converged and e < best['e'] - 1e-7:
                best_mf, best = m, dict(rec, e=e, e_scf=float(m.e_tot), guess=f'{best["guess"]}+stability',
                                        converged=True)
        except Exception as ex:  # noqa: BLE001
            trials.append(dict(guess='stability', error=repr(ex)[:200]))
    # Last resort for finite-field property sets: the same molecule at another field strength, whose stored orbitals
    # reached the published state (all field points of a finite difference must be in the same state; Pol130_O31+
    # stayed closed-shell, +3.5 kcal/mol, while its six field siblings matched Q-Chem, 2026-10-21).
    for sib in _field_siblings(spec['name'])[:3]:
        if best['e'] - e_target <= 1e-5:
            break
        try:
            import cupy as cp
            import h5py
            with h5py.File(ref_path(sib), 'r') as f:
                mo, occ = f['mo_coeff'][:], f['mo_occ'][:]
            if mo.ndim == 2:            # restricted sibling: α = β = half the density
                mo, occ = np.stack([mo, mo]), np.stack([occ / 2, occ / 2])
            dm0 = cp.asarray(np.einsum('spi,si,sqi->spq', mo, occ, mo))
            m = make_mf(mol, spec, True, density_fit=use_df(mol, spec))
            _kernel(m, dm0)
            e = exact_energy(mol, spec, True, m.make_rdm1(), grids_from=m) if use_df(mol, spec) else float(m.e_tot)
            trials.append(dict(guess=f'sibling:{sib}', e=e, converged=bool(m.converged)))
            if m.converged and e < best['e'] - 1e-7:
                best_mf, best = m, dict(rec, e=e, e_scf=float(m.e_tot), guess=f'sibling:{sib}', converged=True)
        except Exception as ex:  # noqa: BLE001
            trials.append(dict(guess=f'sibling:{sib}', error=repr(ex)[:200]))
    best = dict(best, state_search=[{k: v for k, v in t.items() if k != 'log'} | (
        {'stability_rounds': len(t['log'])} if 'log' in t else {}) for t in trials])
    return best_mf, best


STATE_TOL = 1e-4                  # Eh above the published ωB97M-V energy that triggers a state search
STATE_SEARCH_MAX_NAO = int(os.environ.get('GSCDB_STATE_SEARCH_MAX_NAO', '800'))


def reference(mol, spec, e_published=None):
    """ωB97M-V SCF (see module docstring). Returns (mf, record)."""
    closed = spec['mult'] == 1
    if spec['guess_mix']:
        mf, rec = _scf(mol, spec, True, mix=spec['guess_mix'])
        rec['guess'] = f'mix{spec["guess_mix"]}'
    else:
        mf, rec = _scf(mol, spec, not closed)
        rec['guess'] = 'minao'
        if spec['stability'] and closed:
            mf2, rec2 = _scf(mol, spec, True, mix=3)
            rec['alt_bs_e'] = rec2['e']
            if rec2['converged'] and rec2['e'] < rec['e'] - 1e-6:
                mf, rec = mf2, dict(rec2, guess='mix3 (stability)', alt_rks_e=rec['e'])
    if e_published is not None and rec['e'] - e_published > STATE_TOL and mol.nao <= STATE_SEARCH_MAX_NAO:
        e0 = rec['e']
        mf, rec = _state_search(mol, spec, mf, rec, e_published)
        rec['e_before_state_search'] = e0
    rec['unrestricted'] = mf.istype('UKS')
    return mf, rec


def candidate(mol, spec, func, params, ref_mf):
    """Non-SCF energy at the ωB97M-V density, then SCF from the ωB97M-V orbitals."""
    unr = ref_mf.istype('UKS')
    dm_ref = ref_mf.make_rdm1()
    t = time.time()
    e_non = exact_energy(mol, spec, unr, dm_ref, func, params, ref_mf)
    t_non = round(time.time() - t, 1)
    mf, rec = _scf(mol, spec, unr, func, params, dm0=dm_ref)
    rec.update(e_nonscf=e_non, t_nonscf=t_non)
    return mf, rec


def candidate_nonscf(mol, spec, func, params, ref_mf):
    """Non-SCF energy only, at the ωB97M-V density (exact integrals); `converged` is None (no SCF was run)."""
    unr = ref_mf.istype('UKS')
    t = time.time()
    e_non = exact_energy(mol, spec, unr, ref_mf.make_rdm1(), func, params, ref_mf)
    return dict(e_nonscf=e_non, t_nonscf=round(time.time() - t, 1), converged=None, nonscf_only=True)


# ============================================================================ reference orbital store
def ref_path(name, ref_dir=None):
    return os.path.join(ref_dir or G.GSCDB_REF_DIR, f'{name}.h5')


def save_reference(name, mf, rec, ref_dir=None):
    import h5py
    p = ref_path(name, ref_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('mo_coeff', data=mf.mo_coeff.get(), compression='gzip', compression_opts=1)
        f.create_dataset('mo_occ', data=mf.mo_occ.get())
        for k, v in rec.items():
            if isinstance(v, (int, float, str, bool)):
                f.attrs[k] = v
    os.replace(tmp, p)


def load_reference(mol, spec, ref_dir=None):
    """Rebuild a converged ωB97M-V mean-field object from stored orbitals (no SCF). Returns (mf, record) or None."""
    import cupy as cp
    import h5py
    p = ref_path(spec['name'], ref_dir)
    if not os.path.exists(p):
        return None
    with h5py.File(p, 'r') as f:
        mo_c, mo_o = f['mo_coeff'][:], f['mo_occ'][:]
        rec = {k: (v.item() if hasattr(v, 'item') else v) for k, v in f.attrs.items()}
    mf = make_mf(mol, spec, bool(rec.get('unrestricted')), density_fit=use_df(mol, spec))
    mf.mo_coeff, mf.mo_occ = cp.asarray(mo_c), cp.asarray(mo_o)
    mf.converged = bool(rec.get('converged'))
    rec['loaded'] = True
    return mf, rec
