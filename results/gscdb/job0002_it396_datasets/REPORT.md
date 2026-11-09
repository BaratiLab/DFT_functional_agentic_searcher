# GSCDB, 36 sets ≤ 500 basis functions: non-SCF evaluation of iteration 396

Job 0002, label it396. Written 2026-10-23. Prompt: `IT396_GSCDB_LE500_NONSCF_PROMPT.md` in this folder (the comparison
below was fixed there before the run). Baseline: `../job0001_it159_datasets/REPORT.md`. Reporting only: nothing was
fitted, reselected or tuned on these results.

## 1. Setup

| item | value |
|---|---|
| job | 2, unit `gscdb-job0002`, `runs/job0002_it396_datasets/` |
| dates | 2026-10-23 09:55 → 12:23 (2.47 h wall) |
| GPU time | 7.4 GPU-h (prompt estimate 8–15) |
| molecules | 3,487; 3,486 used; AE11_Rn excluded as in job 0001; 0 errors |
| mode | non-SCF only: it396 evaluated on the stored ωB97M-V densities of job 0001 (all 3,486 loaded from the store; reference energies identical to job 0001) |
| functional | it396 `H_388_OSShellELF`, rsh family, 10 parameters (path-fit set, a_s 2.78 / a_o −1.97); MGCDB84 non-SCF val 3.138, test 3.841 |

Non-SCF mode is new: `--nonscf-only` or a `NONSCF_ONLY` file in the job folder. It was added with the user's go-ahead
in GSCDB-only files (`scripts/gscdb_scf.py`, `dftsr/gscdb_scf.py::candidate_nonscf`, `dftsr/gscdb_jobs.py::finalize`;
backups in `code_backup_pre_nonscf/`). A test with it159 reproduced job 0001's non-SCF energies to 9e-12 kcal/mol.
Per molecule, it396 − it159 non-SCF total energies differ by a median of 0.96 kcal/mol (max
39, systematic shifts on heavy atoms that largely cancel in reaction energies).

## 2. Primary result: mean NER over the 35 sets

**it396 does not beat it159.** it396 non-SCF 1.110 vs it159 non-SCF 1.101 (it159 SCF 1.099).
it396 ranks **3** among the 16 published functionals, the same rank as it159.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.930 |
| 2 | ωB97M-V | 1.077 |
| 3 | ωB97M-V (this work, SCF) | 1.086 |
| 4 | it159 SCF | 1.099 |
| 5 | it159 non-SCF | 1.101 |
| 6 | **it396 non-SCF** | 1.110 |
| 7 | ωB97X-V | 1.197 |
| 8 | CF22D | 1.233 |
| 9 | M062X | 1.422 |

## 3. New sets vs sets that overlap MGCDB84

**19 new sets (the transferability group).** it396 1.258 vs it159 non-SCF 1.204: it396 is
worse, and this is the one significant result (§6). Rank 4.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.953 |
| 2 | ωB97X-V | 1.155 |
| 3 | ωB97M-V | 1.186 |
| 4 | ωB97M-V (this work, SCF) | 1.191 |
| 5 | it159 SCF | 1.202 |
| 6 | it159 non-SCF | 1.204 |
| 7 | **it396 non-SCF** | 1.258 |
| 8 | revDSD-PBEP86-D4 | 1.311 |

**16 MGCDB84-derived sets.** it396 0.935 vs it159 non-SCF 0.979: it396 is better and ranks
**2** (behind only ωB97M2).

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.903 |
| 2 | **it396 non-SCF** | 0.935 |
| 3 | ωB97M-V | 0.947 |
| 4 | ωB97M-V (this work, SCF) | 0.961 |
| 5 | it159 SCF | 0.976 |
| 6 | it159 non-SCF | 0.979 |

By MGCDB84 role:

| role | sets | it396 non-SCF | it159 non-SCF | ωB97M-V (ours) | rank of it396 |
|---|---|---|---|---|---|
| train | 5 | 0.913 | 0.985 | 1.013 | 2 |
| validation | 10 | 0.960 | 1.006 | 0.980 | 3 |
| test | 1 | 0.794 | 0.682 | 0.517 | 3 |

The pattern is that of selection on MGCDB84. it396 has the best MGCDB84 validation score of all iterations, and it
gains on the sets that MGCDB84 contains (mean ΔNER -0.044 over the 16; −0.10 to −0.20 on TAE_W4-17MR,
ISOMERIZATION20, BDE99nonMR, G21EA, HAT707nonMR). It loses on the one MGCDB84 test set (WCPT6, +0.11) and on the new
sets (mean ΔNER +0.054; +0.029 without DAPD). This matches the MGCDB84 test split, where
it396 (3.841) was also behind it159 (3.818).

## 4. Mean NER per data type (35 sets)

| data type | sets | it396 non-SCF | it159 non-SCF | it159 SCF | ωB97M-V (ours) | ωB97M-V (published) |
|---|---|---|---|---|---|---|
| Barrier Height | 2 | 1.022 | 1.038 | 1.038 | 1.057 | 0.972 |
| Electric field | 3 | 1.472 | 1.457 | 1.454 | 1.398 | 1.403 |
| Frequency | 1 | 1.139 | 1.117 | 1.116 | 0.926 | 0.925 |
| Isomerization | 1 | 1.160 | 1.326 | 1.322 | 1.166 | 1.166 |
| Noncovalent | 9 | 0.891 | 0.872 | 0.871 | 0.931 | 0.932 |
| Thermochemistry | 16 | 1.044 | 1.068 | 1.065 | 1.054 | 1.043 |
| Transition Metal | 3 | 1.790 | 1.571 | 1.569 | 1.459 | 1.462 |

it396 is better than it159 on thermochemistry (1.044 vs 1.068) and isomerization (1.160 vs 1.326). It is worse on
transition metals (1.790 vs 1.571; DAPD alone +0.50 NER), electric-field properties and noncovalent sets.

## 5. Per-set table

| set | group | type | n | metric it396 | metric it159 | NER it396 | NER it159 | ΔNER | NER ωB97M-V (ours) | winner (20 methods) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3d4dIPSS | new | Transition Metal | 32 | 6.2156 | 5.6159 | 1.197 | 1.081 | +0.116 | 0.977 | revDSD-PBEP86-D4 |
| AE18 | train | Thermochemistry | 18 | 4.6767 | 4.9247 | 0.944 | 0.995 | -0.051 | 1.072 | M062X |
| ALKBDE10 | new | Thermochemistry | 10 | 3.7128 | 3.5683 | 1.141 | 1.096 | +0.045 | 1.100 | MN15 |
| BDE99MR | validation | Thermochemistry | 16 | 4.3852 | 4.2224 | 1.323 | 1.274 | +0.049 | 1.228 | ωB97M2 |
| BDE99nonMR | train | Thermochemistry | 83 | 1.2743 | 1.5563 | 0.660 | 0.806 | -0.146 | 0.790 | ωB97M2 |
| BH76RC | validation | Thermochemistry | 30 | 0.9653 | 0.9024 | 0.959 | 0.897 | +0.062 | 0.846 | ωB97M2 |
| CT20 | validation | Noncovalent | 20 | 0.0717 | 0.0687 | 0.803 | 0.768 | +0.035 | 0.818 | it159 non-SCF |
| DAPD | new | Transition Metal | 12 | 8.0299 | 6.6315 | 2.892 | 2.388 | +0.504 | 2.256 | PW6B95 |
| DBH22 | train | Barrier Height | 22 | 1.0767 | 1.1511 | 0.892 | 0.954 | -0.062 | 0.973 | ωB97M2 |
| DIPCS9 | new | Thermochemistry | 9 | 4.0819 | 4.1831 | 1.207 | 1.237 | -0.030 | 1.510 | ωB97M2 |
| Dip146 | new | Electric field | 190 | 0.0402 | 0.0392 | 1.155 | 1.128 | +0.027 | 1.051 | ωB97M2 |
| G21EA | train | Thermochemistry | 25 | 1.2030 | 1.3434 | 0.879 | 0.982 | -0.103 | 1.001 | it396 non-SCF |
| HAT707MR | validation | Thermochemistry | 202 | 3.5115 | 3.5782 | 1.093 | 1.114 | -0.021 | 1.090 | ωB97M2 |
| HAT707nonMR | validation | Thermochemistry | 505 | 1.5889 | 1.8178 | 0.672 | 0.769 | -0.097 | 0.805 | ωB97M2 |
| HCP32 | new | Noncovalent | 32 | 0.4990 | 0.5144 | 0.872 | 0.899 | -0.027 | 0.879 | ωB97X-V |
| HEAVY28 | new | Noncovalent | 28 | 0.1653 | 0.1554 | 0.874 | 0.822 | +0.052 | 0.943 | revDSD-PBEP86-D4 |
| He3 | new | Noncovalent | 49 | 0.1766 | 0.1715 | 1.002 | 0.973 | +0.029 | 1.038 | revDSD-PBEP86-D4 |
| IP23 | new | Thermochemistry | 23 | 2.2011 | 2.4599 | 0.866 | 0.968 | -0.102 | 0.991 | ωB97M2 |
| ISOMERIZATION20 | validation | Isomerization | 20 | 1.5169 | 1.7345 | 1.160 | 1.326 | -0.166 | 1.166 | ωB97M2 |
| MB08-165 | new | Thermochemistry | 165 | 4.2090 | 4.0085 | 0.920 | 0.876 | +0.044 | 0.829 | ωB97M-V |
| NC11 | validation | Noncovalent | 11 | 0.0415 | 0.0413 | 0.763 | 0.759 | +0.004 | 0.829 | revDSD-PBEP86-D4 |
| O24 | new | Noncovalent | 24 | 1.0286 | 0.9083 | 0.915 | 0.808 | +0.107 | 0.914 | revDSD-PBEP86-D4 |
| O24x4 | new | Noncovalent | 96 | 0.6132 | 0.5830 | 0.875 | 0.831 | +0.044 | 0.860 | ωB97M2 |
| OEEF | new | Electric field | 136 | 0.0939 | 0.1029 | 1.072 | 1.174 | -0.102 | 1.134 | ωB97M2 |
| ORBH35 | new | Barrier Height | 35 | 2.6117 | 2.5432 | 1.153 | 1.122 | +0.031 | 1.140 | CF22D |
| P34EA | new | Thermochemistry | 9 | 3.1748 | 3.0067 | 2.022 | 1.915 | +0.107 | 1.777 | ωB97M2 |
| Pol130 | new | Electric field | 296 | 0.0520 | 0.0491 | 2.189 | 2.068 | +0.121 | 2.010 | r2SCAN0 |
| RG10N | validation | Noncovalent | 275 | 0.0772 | 0.0847 | 0.727 | 0.797 | -0.070 | 0.872 | ωB97M2 |
| SIE4x4 | new | Thermochemistry | 16 | 10.3673 | 10.3683 | 1.130 | 1.130 | +0.000 | 1.156 | revDSD-PBEP86-D4 |
| SN13 | validation | Thermochemistry | 13 | 0.5570 | 0.5915 | 0.883 | 0.937 | -0.054 | 0.799 | ωB97M-V |
| TA13 | train | Noncovalent | 13 | 1.6129 | 1.6131 | 1.189 | 1.189 | +0.000 | 1.228 | revDSD-PBEP86-D4 |
| TAE_W4-17MR | validation | Thermochemistry | 17 | 3.7392 | 4.3500 | 1.216 | 1.414 | -0.198 | 1.345 | PW6B95 |
| TMD10 | new | Transition Metal | 10 | 7.2367 | 7.0160 | 1.282 | 1.243 | +0.039 | 1.144 | CF22D |
| V30 | new | Frequency | 275 | 34.2388 | 33.5727 | 1.139 | 1.117 | +0.022 | 0.926 | revDSD-PBEP86-D4 |
| WCPT6 | test | Thermochemistry | 6 | 0.5265 | 0.4525 | 0.794 | 0.682 | +0.112 | 0.517 | ωB97M-V |

All NER and metric values for it396 and it159 here are non-SCF. AE11 is not compared (AE11_Rn excluded).

## 6. Paired comparison

ΔNER = NER(it396 non-SCF) − NER(other) per set; Wilcoxon signed-rank test; bootstrap 95 % CI of the mean ΔNER
(10,000 draws, seed 0).

| sets | vs | mean ΔNER | median ΔNER | it396 better | Wilcoxon p | bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| all 35 | it159 non-SCF | +0.0091 | +0.0222 | 16/35 | 0.777 | [-0.0267, +0.0494] |
| all 35 | ωB97M-V (this work, SCF) | +0.0240 | -0.0058 | 18/35 | 0.790 | [-0.0277, +0.0807] |
| all 35 | it159 SCF | +0.0112 | +0.0230 | 15/35 | 0.765 | [-0.0246, +0.0515] |
| 19 new | it159 non-SCF | +0.0539 | +0.0391 | 5/19 | 0.018 | [+0.0073, +0.1164] |
| 19 new | ωB97M-V (this work, SCF) | +0.0666 | +0.0142 | 7/19 | 0.169 | [-0.0134, +0.1584] |
| 19 new | it159 SCF | +0.0557 | +0.0414 | 5/19 | 0.020 | [+0.0083, +0.1186] |

- Over all 35 sets it396 and it159 are indistinguishable (p ≈ 0.78).
- On the 19 new sets it396 is significantly worse than it159: p 0.018 against it159 non-SCF and p 0.020 against
  it159 SCF, with bootstrap CIs that exclude zero.
- Against our ωB97M-V no difference is significant.

## 7. Caveats

- Same scope as job 0001: a size-restricted subset (35 of the 137 GSCDB sets, AE11 excluded). The subset ranking agrees only
  partly with the full GSCDB ranking (Spearman 0.93, Kendall 0.78; for close pairs the subset picked the other winner
  in 4 of 10 cases).
- Non-SCF only. For it159, SCF and non-SCF differed by ≤ 0.004 NER in every table, so this comparison is very likely
  representative, but no it396 SCF was run.
- Both functionals use the same reference densities and electronic states (job 0001). The state differences against
  Q-Chem listed there (O₃, CO₃, C₂, a few 3d/4d atoms) affect it396 and it159 alike.
- Two of the six tests are significant, and both are the same comparison: it396 vs it159 (non-SCF and SCF) on the new sets.
  With six tests, p ≈ 0.02 is suggestive rather than decisive. The direction agrees with the MGCDB84 test split.
- it396 was chosen on MGCDB84 criteria only, before this run. These results must not be used to pick or tune a
  functional.

## 8. Next steps (for the user)

- No it396 SCF run is suggested. it396 does not win overall, and it is worse on the new sets.
- it159 remains the reference candidate on this subset. The open options from job 0001 still apply: GAS22 (now about
  7–8 GPU-h in non-SCF mode, or about 50 GPU-h with SCF), an AE11_Rn CPU-XC fallback, and the extension to ≤ 1000 bf.

## Addendum (2026-10-24): AE11_Rn added, all 3,487 molecules

AE11_Rn could not be computed on the GPU (GPU4PySCF semilocal-XC defect for all-electron Rn). It was computed on the CPU for
it396 non-SCF with `runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py`, from the stored CPU reference orbitals (the reference is
within 0.003 kcal/mol of Q-Chem; the script was validated on AE11_Kr to 5e-6 kcal/mol). `molecules.jsonl`, `result.json` and
`energies.csv` now cover 3,487 molecules; the old files are `result_before_AE11_Rn.json` and `energies_before_AE11_Rn.csv`.
Everything above is the 35-set analysis and is unchanged. **36 sets (AE11 included):** it396 non-SCF 1.147 (rank 3 of 16). it396 − it159 non-SCF +0.006 (p 0.93); on the 20 new sets +0.046 (p 0.048). AE11 NER: it396 2.43.
Full 36-set tables and paired tests: `runs/job0004_GAS22_SCF_datasets/REPORT.md` §11.

## Addendum (2026-10-24): convergence fix

The two unconverged ωB97M-V references were converged on the CPU and replaced; it396 non-SCF of those two molecules recomputed (change < 1e-5 kcal/mol). Mean NER unchanged. Details: `runs/convergence_fix/README.md` and `runs/job0004_GAS22_SCF_datasets/REPORT.md` §12. The numbers above are those before the fix, where they differ.
Before-fix files: `molecules_before_convergence_fix.jsonl`, `result_before_convergence_fix.json`.
