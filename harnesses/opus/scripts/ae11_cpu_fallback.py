#!/usr/bin/env python
"""CPU-XC fallback for GSCDB molecules that GPU4PySCF 1.8.1 cannot do (all-electron Rn, exponent 2.3e8).

Standalone: it imports the project's `dftsr` modules but does not change them. The whole SCF runs on the CPU with
PySCF (exact in-core J/K for the full-range and the ω = 0.3 long-range ERIs, libxc ωB97M-V or the candidate's
PyTorch semilocal XC, VV10 on the CPU grids), with the same grids, convergence thresholds and SCF fallbacks as
`dftsr/gscdb_scf.py`: reference ωB97M-V SCF -> candidate non-SCF on that density -> candidate SCF from those orbitals.

  python ae11_cpu_fallback.py --mol AE11_Kr --out DIR --check      # validation: compare with the GPU records, write nothing to the store
  python ae11_cpu_fallback.py --mol AE11_Rn --out DIR --write      # compute, store the reference orbitals, append the record
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pyscf import dft, lib
from pyscf.scf import hf as cpu_hf
from dftsr import gscdb as G
from dftsr.gscdb_scf import CONV_TOL

RUNS = G.GSCDB_RUNS + os.sep


def load_records(path):
    d = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); d[r['name']] = r
    return d


def cpu_mf(mol, spec, numint=None, eris=None):
    mf = dft.RKS(mol, xc='wb97m-v')
    if numint is not None:
        mf._numint = numint
    mf.grids.atom_grid = spec['xc_grid']
    mf.grids.prune = None
    mf.nlcgrids.atom_grid = (50, 194) if spec['nl_grid'] == 'sg1' else spec['nl_grid']
    mf.nlcgrids.prune = None                      # Z > 18: the GPU code does the same
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = 10.0 ** -(spec['scf_convergence'] - 1)
    mf.max_cycle = 150
    mf.max_memory = 300000
    full, lr = eris

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        return cpu_hf.dot_eri_dm(lr if omega else full, dm, hermi, with_j, with_k)
    mf.get_jk = get_jk
    return mf


def kernel(mf, dm0=None):
    """Same fallbacks as gscdb_scf._kernel: DIIS -> level shift + damping -> Newton."""
    mf.kernel(dm0=dm0)
    how = 'diis'
    if not mf.converged:
        mf.level_shift, mf.damp, mf.max_cycle = 0.25, 0.5, 300
        mf.kernel(dm0=mf.make_rdm1())
        how = 'level_shift'
        mf.level_shift, mf.damp = 0.0, 0.0
    if not mf.converged:
        try:
            mf2 = mf.newton(); mf2.max_cycle = 100
            mf2.kernel(mf.mo_coeff, mf.mo_occ)
            if mf2.converged:
                mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot = mf2.mo_coeff, mf2.mo_occ, mf2.mo_energy, mf2.e_tot
                mf.converged = True
            how = 'newton'
        except Exception as e:  # noqa: BLE001
            how = f'newton_failed:{type(e).__name__}'
    return how


def scf(mf, dm0=None):
    t = time.time()
    how = kernel(mf, dm0)
    rec = dict(converged=bool(mf.converged), how=how, cycles=getattr(mf, 'cycles', None), e_scf=float(mf.e_tot),
               df=False, t_scf=round(time.time() - t, 1))
    rec['e'] = rec['e_scf']                       # exact JK: the SCF energy is the exact-integral energy
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--candidate', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'baselines', 'gas22.py'))
    ap.add_argument('--params', default='')
    ap.add_argument('--label', default='GAS22')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--skip-candidate', action='store_true')
    ap.add_argument('--ref-from-store', action='store_true',
                    help='use the stored reference orbitals (the CPU ones written by --write) instead of a new reference SCF')
    ap.add_argument('--nonscf-only', action='store_true', help='candidate non-SCF only (jobs run in NONSCF_ONLY mode)')
    ap.add_argument('--record-out', default='', help='write the record (json) here; nothing else is written')
    a = ap.parse_args()
    t0 = time.time()
    spec = G.molecule_spec(a.mol)
    mol = G.build_mol(spec, max_memory=300000)
    assert spec['mult'] == 1 and not spec['field'] and not spec['guess_mix'], 'closed-shell, field-free only'
    print(a.mol, 'nao', mol.nao, flush=True)
    tt = time.time()
    full = mol.intor('int2e', aosym='s8')
    with mol.with_range_coulomb(0.3):
        lr = mol.intor('int2e', aosym='s8')
    print('ERIs', round(time.time() - tt, 1), 's', full.nbytes / 1e9, 'GB each', flush=True)
    eris = (full, lr)

    # ---- reference: libxc ωB97M-V
    mf = cpu_mf(mol, spec, eris=eris)
    if a.ref_from_store:
        import h5py
        with h5py.File(os.path.join(G.GSCDB_REF_DIR, f'{a.mol}.h5'), 'r') as f:
            mf.mo_coeff, mf.mo_occ = f['mo_coeff'][:], f['mo_occ'][:]
            ref = {k: (v.item() if hasattr(v, 'item') else v) for k, v in f.attrs.items()}
        mf.converged = bool(ref.get('converged'))
        ref.update(loaded=True, cpu_fallback=True)
    else:
        ref = scf(mf)
        ref.update(guess='minao', unrestricted=False, loaded=False, cpu_fallback=True)
    pub = G.published_energies()['wB97M-V']
    print('reference', ref['converged'], ref['how'], ref['cycles'], ref['e'], 'published', pub.get(a.mol),
          'diff kcal', (ref['e'] - pub[a.mol]) * G.HARTREE_TO_KCAL, flush=True)
    rec = dict(name=a.mol, gpu=-1, nao=mol.nao, mult=spec['mult'], df=False, ref=ref, E_ref=ref['e'],
               E_ref_published=pub[a.mol], dE_ref_published=ref['e'] - pub[a.mol], cpu_fallback=True)
    dm_ref = mf.make_rdm1()

    # ---- candidate: non-SCF on the reference density, then SCF from the reference orbitals
    if not a.skip_candidate:
        import cupy as cp
        from pyscf.dft import numint
        from dftsr.api import load_candidate
        from dftsr.scf import TorchNumInt
        func = load_candidate(a.candidate)
        params = json.load(open(a.params or os.path.join(a.out, 'params.json')))
        tni = TorchNumInt(func, params)

        class CPUTorchNumInt(numint.NumInt):
            def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None, spin=None):
                rho = np.asarray(rho, order='C', dtype=np.double)
                if xctype == 'MGGA' and rho.shape[-2] == 6:
                    rho = np.asarray(rho[..., [0, 1, 2, 3, 5], :], order='C')
                spin = 1 if (rho.ndim == 3 and rho.shape[0] == 2) else 0
                exc, vxc, _, _ = tni.eval_xc_eff(xc_code, cp.asarray(rho), deriv, omega=omega, xctype=xctype, spin=spin)
                return [exc.get(), vxc.get(), None, None]

        mfc = cpu_mf(mol, spec, numint=CPUTorchNumInt(), eris=eris)
        tn = time.time()
        e_non = float(mfc.energy_tot(dm=dm_ref))
        t_non = round(time.time() - tn, 1)
        if a.nonscf_only:
            c = dict(e_nonscf=e_non, t_nonscf=t_non, converged=None, nonscf_only=True)
            print('candidate non-SCF', e_non, flush=True)
            rec.update(cand=c, E_nonscf=e_non)
        else:
            c = scf(mfc, dm0=dm_ref)
            c.update(e_nonscf=e_non, t_nonscf=t_non)
            print('candidate non-SCF', e_non, 'SCF', c['e'], c['converged'], c['how'], c['cycles'], flush=True)
            rec.update(cand=c, E_nonscf=e_non, E_scf=c['e'], converged=c['converged'])
    rec['status'] = 'ok'
    rec['t'] = round(time.time() - t0, 1)

    if a.check:
        for jn, lab in (('job0004_GAS22_SCF_datasets', 'job4'), ('job0001_it159_datasets', 'job1')):
            g = load_records(RUNS + jn + '/molecules.jsonl').get(a.mol)
            if g and 'E_ref' in g:
                print(lab, 'GPU E_ref', g['E_ref'], 'CPU-GPU kcal', (rec['E_ref'] - g['E_ref']) * G.HARTREE_TO_KCAL)
                if lab == 'job4' and 'cand' in rec:
                    print(lab, 'GPU E_nonscf', g['E_nonscf'], 'diff kcal', (rec['E_nonscf'] - g['E_nonscf']) * G.HARTREE_TO_KCAL)
                    print(lab, 'GPU E_scf', g['E_scf'], 'diff kcal', (rec['E_scf'] - g['E_scf']) * G.HARTREE_TO_KCAL)
    if a.record_out:
        json.dump(rec, open(a.record_out, 'w'))
        print('record written', a.record_out, flush=True)
    if a.write:
        import h5py
        p = os.path.join(G.GSCDB_REF_DIR, f'{a.mol}.h5')
        if os.path.exists(p):
            shutil.copy(p, os.path.join(a.out, f'{a.mol}.h5.before_cpu_fallback'))      # backup of the stored reference
        tmp = p + '.tmp'
        with h5py.File(tmp, 'w') as f:
            f.create_dataset('mo_coeff', data=mf.mo_coeff, compression='gzip', compression_opts=1)
            f.create_dataset('mo_occ', data=mf.mo_occ)
            for k, v in ref.items():
                if isinstance(v, (int, float, str, bool)):
                    f.attrs[k] = v
        os.replace(tmp, p)
        with open(os.path.join(a.out, 'ae11_cpu_fallback_record.json'), 'w') as f:
            json.dump(rec, f)
        print('stored reference + record json written', flush=True)
    print('total', rec['t'], 's')


if __name__ == '__main__':
    main()
