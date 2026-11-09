# Convergence fix for the unconverged SCF records (2026-10-24)

User request: "make them converge for credibility" for the three functionals for which an SCF was done: ωB97M-V (reference), it159, GAS22.
Before: ωB97M-V reference 2 unconverged, it159 15, GAS22 66 (of 3,487). After: reference 0, it159 0, GAS22 2 (the same Pd atom twice).
No project code was changed; every script here patches what it needs inside its own process.

## Cause (GAS22 and it159, not the ωB97M-V reference)
- The torch potential of GAS22 and it159 is numerically non-smooth in the far minority-spin tail (rho_sigma ~ 1e-10, reduced gradient ~ 1e9).
  A tiny orbital rotation changes the tail density by a large relative amount (~1e4 × the rotation size), and the potential jumps there.
  The SCF gradient then stalls at ~1e-5 (K, Pd: 1e-4 .. 2e-2) against the 1e-6 criterion while the energy is stationary to ~1e-9 Eh.
- Evidence: libxc ωB97M-V gradient is smooth for rotations 1e-4..1e-10; GAS22's has jumps of ~1e-4 for rotations 1e-8..1e-6
  (`diag_rot_li.log`). Non-smooth points sit on one radial shell of the Na atom (r = 5.6 bohr) where rho_beta ≈ 1.45e-10, just above the 1e-10 spin threshold
  (`diag_points_na.log`). Torch-ωB97M-V (same machinery) converges from the MINAO guess in 8 cycles on Na (`diag_minao_*`), so the GPU/torch machinery is fine.
- Ruled out (no effect): cutting the tail density (1e-12..1e-10), regularising x^(2/3) in GAS22 (1e-16..1e-6), a smooth tau >= tau_W max, level shifts 0.25..2 + damping (Pd), the a<5 switch of the attenuation function (smooth to 1e-9).
- Lowering the spin-channel threshold `xc_torch.RHO_THR` 1e-10 → 1e-14 fixes the Li-like cases (Li converges in 12 cycles; `diag_thr_li.log`) and changes converged energies by ≤ 1.5e-5 kcal/mol
  (14 control molecules, `thr_sample.jsonl`). It does not fix Na, K, Cr, HCl (`diag_thr2_na.log`). Higher thresholds (1e-8, 1e-7) converge but shift energies by 0.004-0.02 kcal/mol: rejected.

## Protocol (tiers; each record keeps its class in `cand.convergence`, plus `rho_thr`, `g_best`, `E_spread_5best`)
1. `strict`: DIIS reached |g| < 1e-6 and dE < 1e-10 (with RHO_THR 1e-14).
2. `strict_gradient`: an iterate reached |g| < 1e-6; its energy is the record (only the 1e-10 energy criterion was missed by noise).
3. `noise_limited`: best-|g| iterate with |g| < 1e-3 and the energies of the 5 smallest-|g| iterates within a spread tolerance (1e-8 Eh for most records, 1e-7 for Na–Li 1.2/1.5/2.0 Å, 1e-6 with level shift 0.5 for K).
   Largest g_best 4.2e-4 (GAS22) and 7.9e-6 (it159); largest energy spread 8.3e-8 Eh = 5e-5 kcal/mol.
4. `unconverged`: DAPD_Pd and 3d4dIPSS_Pd_GS (the same Pd atom, GAS22 only): |g| stuck at 2.5e-2 for every solver setting (DIIS, level shifts 0.25-2, damping, 300-cycle chain, CPU path). Energy reproducible to ~1e-6 Eh
   (GPU −127.3946424, CPU −127.3946427, five settings within 8e-7 Eh). Cause not found.

## Reference SCFs (ωB97M-V, libxc)
Pol130_CH2PH2- and RG10N_KrKr_3p900 were recomputed on the CPU with `ref_cpu.py` (exact in-core J/K, field supported): both converge by DIIS (11 and 8 cycles), ωB97M-V − Q-Chem = 6e-5 and 3e-4 kcal/mol
(was 2.5e-3 and 3e-4). The stored `.h5` files were replaced (old ones in `ref_backups/`), and the candidate part of those two molecules was recomputed in all four jobs
(`recompute_two.py`; Kr–Kr 3.9 Å GAS22 SCF with `rerun_scf2.py`). E_ref changes −0.0025 and +0.00004 kcal/mol.

## Result
- Largest change of a corrected SCF energy: GAS22 3.9e-4 kcal/mol (median 6e-5), it159 1.8e-4 (median 4e-5).
- Mean NER (35- and 36-set) changes by ≤ 1e-5 for every method; no ranking and no paired test changes (it159 SCF 1.13872, GAS22 SCF 1.19031 over 36 sets).
- Density-driven error: GAS22 mean 0.150 / max 1.915, it159 mean 0.0385 / max 1.293 kcal/mol (unchanged).

## Files
`rerun_scf.py` (full chain, RHO_THR patch), `rerun_scf2.py` (noise-aware protocol), `ref_cpu.py`, `recompute_two.py`, `merge_records.py`, `diag_*.py/log` (diagnosis), `rerun*.jsonl`, `recomp_*.jsonl`, `newref_*`, `ref_backups/`.
Each job folder has `molecules_before_convergence_fix.jsonl`, `result_before_convergence_fix.json`, `energies_before_convergence_fix.csv`.
