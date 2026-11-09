# GSCDB, 36 sets ≤ 500 basis functions: non-SCF evaluation of GAS22

Job 0003, label GAS22 (published parameters). Written 2026-10-23. Prompt with the pre-registered comparison:
`GAS22_GSCDB_LE500_NONSCF_PROMPT.md` in this folder. Baselines: `../job0001_it159_datasets/REPORT.md`,
`../job0002_it396_datasets/REPORT.md`. Reporting only.

## 1. Setup

| item | value |
|---|---|
| job | 3 (registered in `gscdb_jobs.sqlite`, iteration_id 0), unit `gscdb-job0003`, launched with systemd-run because GAS22 is not an MCP iteration |
| dates | 2026-10-23 12:41 → 15:07 (2.43 h wall) |
| GPU time | 7.3 GPU-h |
| molecules | 3,487; 3,486 used; AE11_Rn excluded as before; 0 errors |
| mode | non-SCF on the stored ωB97M-V densities of job 0001 (all loaded; reference energies identical) |
| functional | GAS22 (Ma et al., Sci. Adv. 2022), `baselines/gas22.py`, published parameters; ωB97M-V frame (ω 0.3, 0.15 SR-HF, VV10); MGCDB84 non-SCF val 3.462, test 3.600 (best test of all functionals evaluated in this project) |

GAS22's total energies differ strongly from it159's for heavy all-electron atoms (median |Δ| per molecule
6.1 kcal/mol; Hg −888, Yb −645, Ba −365, Xe −335, Kr −236 kcal/mol). The shift is systematic.
It grows with Z, is the same within a series (all Kr–Kr dimers ≈ −236) and cancels in reaction energies.
It is consistent with GAS22's exchange not reproducing the uniform-gas limit (c₀ = 0.862). AE11, the only set of
absolute heavy-atom energies, is excluded anyway.

## 2. Primary result: mean NER over the 35 sets

**GAS22 is clearly behind it159.** GAS22 non-SCF 1.228, compared with it159 non-SCF 1.101,
it396 non-SCF 1.110 and our ωB97M-V 1.086. It ranks **4** among the 16 published,
behind ωB97M2, ωB97M-V and ωB97X-V and just ahead of CF22D.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.930 |
| 2 | ωB97M-V | 1.077 |
| 3 | ωB97M-V (this work, SCF) | 1.086 |
| 4 | it159 SCF | 1.099 |
| 5 | it159 non-SCF | 1.101 |
| 6 | it396 non-SCF | 1.110 |
| 7 | ωB97X-V | 1.197 |
| 8 | **GAS22 non-SCF** | 1.228 |
| 9 | CF22D | 1.233 |
| 10 | M062X | 1.422 |

## 3. New sets vs sets that overlap MGCDB84

**19 new sets:** GAS22 1.379 (rank 7), compared with it159 1.204 and it396 1.258.
**16 MGCDB84-derived sets:** GAS22 1.047 (rank 3), compared with it159 0.979 and it396 0.935.

| role | sets | GAS22 | it159 non-SCF | it396 non-SCF | ωB97M-V (ours) | rank of GAS22 |
|---|---|---|---|---|---|---|
| train | 5 | 1.034 | 0.985 | 0.913 | 1.013 | 4 |
| validation | 10 | 1.100 | 1.006 | 0.960 | 0.980 | 4 |
| test | 1 | 0.584 | 0.682 | 0.794 | 0.517 | 2 |

GAS22 is behind on both groups. The one exception is WCPT6, the single MGCDB84 test-role set, where GAS22 (0.584)
beats it159 (0.683) and it396 (0.794).

## 4. Mean NER per data type (35 sets)

| data type | sets | GAS22 | it159 non-SCF | it396 non-SCF | ωB97M-V (ours) | ωB97M-V (published) |
|---|---|---|---|---|---|---|
| Barrier Height | 2 | 1.087 | 1.038 | 1.022 | 1.057 | 0.972 |
| Electric field | 3 | 1.572 | 1.457 | 1.472 | 1.398 | 1.403 |
| Frequency | 1 | 1.068 | 1.117 | 1.139 | 0.926 | 0.925 |
| Isomerization | 1 | 1.433 | 1.326 | 1.160 | 1.166 | 1.166 |
| Noncovalent | 9 | 0.861 | 0.872 | 0.891 | 0.931 | 0.932 |
| Thermochemistry | 16 | 1.200 | 1.068 | 1.044 | 1.054 | 1.043 |
| Transition Metal | 3 | 2.204 | 1.571 | 1.790 | 1.459 | 1.462 |

GAS22 is best of the three only on noncovalent sets (0.861) and the V30 frequencies (1.068). It is clearly worse on
transition metals (2.204 vs 1.571 for it159; 3d4dIPSS +1.00 and TMD10 +0.73 NER), on thermochemistry (1.200 vs 1.068;
SN13 +0.62, ALKBDE10 +0.39, P34EA +0.32, MB08-165 +0.29) and on electric-field properties (1.572).

## 5. Per-set table

| set | group | type | n | metric GAS22 | NER GAS22 | NER it159 (non-SCF) | NER it396 (non-SCF) | ΔNER GAS22 − it159 | NER ωB97M-V (ours) | winner (21 methods) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3d4dIPSS | new | Transition Metal | 32 | 10.8218 | 2.084 | 1.081 | 1.197 | +1.003 | 0.977 | revDSD-PBEP86-D4 |
| AE18 | train | Thermochemistry | 18 | 3.3111 | 0.669 | 0.995 | 0.944 | -0.326 | 1.072 | M062X |
| ALKBDE10 | new | Thermochemistry | 10 | 4.8379 | 1.486 | 1.096 | 1.141 | +0.390 | 1.100 | MN15 |
| BDE99MR | validation | Thermochemistry | 16 | 4.3113 | 1.301 | 1.274 | 1.323 | +0.027 | 1.228 | ωB97M2 |
| BDE99nonMR | train | Thermochemistry | 83 | 1.7789 | 0.921 | 0.806 | 0.660 | +0.115 | 0.790 | ωB97M2 |
| BH76RC | validation | Thermochemistry | 30 | 1.1643 | 1.157 | 0.897 | 0.959 | +0.260 | 0.846 | ωB97M2 |
| CT20 | validation | Noncovalent | 20 | 0.0765 | 0.856 | 0.768 | 0.803 | +0.088 | 0.818 | it159 non-SCF |
| DAPD | new | Transition Metal | 12 | 7.0912 | 2.554 | 2.388 | 2.892 | +0.166 | 2.256 | PW6B95 |
| DBH22 | train | Barrier Height | 22 | 1.2540 | 1.039 | 0.954 | 0.892 | +0.085 | 0.973 | ωB97M2 |
| DIPCS9 | new | Thermochemistry | 9 | 4.1815 | 1.236 | 1.237 | 1.207 | -0.001 | 1.510 | ωB97M2 |
| Dip146 | new | Electric field | 190 | 0.0376 | 1.080 | 1.128 | 1.155 | -0.048 | 1.051 | ωB97M2 |
| G21EA | train | Thermochemistry | 25 | 1.7282 | 1.263 | 0.982 | 0.879 | +0.281 | 1.001 | it396 non-SCF |
| HAT707MR | validation | Thermochemistry | 202 | 3.6813 | 1.146 | 1.114 | 1.093 | +0.032 | 1.090 | ωB97M2 |
| HAT707nonMR | validation | Thermochemistry | 505 | 1.9654 | 0.831 | 0.769 | 0.672 | +0.062 | 0.805 | ωB97M2 |
| HCP32 | new | Noncovalent | 32 | 0.5019 | 0.878 | 0.899 | 0.872 | -0.021 | 0.879 | ωB97X-V |
| HEAVY28 | new | Noncovalent | 28 | 0.1314 | 0.695 | 0.822 | 0.874 | -0.127 | 0.943 | revDSD-PBEP86-D4 |
| He3 | new | Noncovalent | 49 | 0.1680 | 0.953 | 0.973 | 1.002 | -0.020 | 1.038 | revDSD-PBEP86-D4 |
| IP23 | new | Thermochemistry | 23 | 2.6333 | 1.037 | 0.968 | 0.866 | +0.069 | 0.991 | ωB97M2 |
| ISOMERIZATION20 | validation | Isomerization | 20 | 1.8741 | 1.433 | 1.326 | 1.160 | +0.107 | 1.166 | ωB97M2 |
| MB08-165 | new | Thermochemistry | 165 | 5.3265 | 1.165 | 0.876 | 0.920 | +0.289 | 0.829 | ωB97M-V |
| NC11 | validation | Noncovalent | 11 | 0.0345 | 0.635 | 0.759 | 0.763 | -0.124 | 0.829 | revDSD-PBEP86-D4 |
| O24 | new | Noncovalent | 24 | 1.0408 | 0.925 | 0.808 | 0.915 | +0.117 | 0.914 | revDSD-PBEP86-D4 |
| O24x4 | new | Noncovalent | 96 | 0.6323 | 0.902 | 0.831 | 0.875 | +0.071 | 0.860 | ωB97M2 |
| OEEF | new | Electric field | 136 | 0.1293 | 1.476 | 1.174 | 1.072 | +0.302 | 1.134 | ωB97M2 |
| ORBH35 | new | Barrier Height | 35 | 2.5731 | 1.136 | 1.122 | 1.153 | +0.014 | 1.140 | CF22D |
| P34EA | new | Thermochemistry | 9 | 3.5060 | 2.232 | 1.915 | 2.022 | +0.317 | 1.777 | ωB97M2 |
| Pol130 | new | Electric field | 296 | 0.0514 | 2.161 | 2.068 | 2.189 | +0.093 | 2.010 | r2SCAN0 |
| RG10N | validation | Noncovalent | 275 | 0.0669 | 0.630 | 0.797 | 0.727 | -0.167 | 0.872 | ωB97M2 |
| SIE4x4 | new | Thermochemistry | 16 | 10.7035 | 1.166 | 1.130 | 1.130 | +0.036 | 1.156 | revDSD-PBEP86-D4 |
| SN13 | validation | Thermochemistry | 13 | 0.9838 | 1.559 | 0.937 | 0.883 | +0.622 | 0.799 | ωB97M-V |
| TA13 | train | Noncovalent | 13 | 1.7338 | 1.278 | 1.189 | 1.189 | +0.089 | 1.228 | revDSD-PBEP86-D4 |
| TAE_W4-17MR | validation | Thermochemistry | 17 | 4.4617 | 1.450 | 1.414 | 1.216 | +0.036 | 1.345 | PW6B95 |
| TMD10 | new | Transition Metal | 10 | 11.1399 | 1.974 | 1.243 | 1.282 | +0.731 | 1.144 | CF22D |
| V30 | new | Frequency | 275 | 32.0930 | 1.068 | 1.117 | 1.139 | -0.049 | 0.926 | revDSD-PBEP86-D4 |
| WCPT6 | test | Thermochemistry | 6 | 0.3875 | 0.584 | 0.682 | 0.794 | -0.098 | 0.517 | ωB97M-V |

## 6. Paired comparison

ΔNER = NER(GAS22 non-SCF) − NER(other) per set; Wilcoxon signed-rank test; bootstrap 95 % CI (10,000 draws, seed 0).

| sets | vs | mean ΔNER | median ΔNER | GAS22 better | Wilcoxon p | bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| all 35 | it159 non-SCF | +0.1263 | +0.0704 | 10/35 | 0.0048 | [+0.0478, +0.2167] |
| all 35 | it396 non-SCF | +0.1172 | +0.0529 | 12/35 | 0.0230 | [+0.0350, +0.2086] |
| all 35 | ωB97M-V (this work, SCF) | +0.1412 | +0.0659 | 8/35 | 0.0028 | [+0.0455, +0.2458] |
| 19 new | it159 non-SCF | +0.1753 | +0.0704 | 6/19 | 0.0108 | [+0.0653, +0.3122] |
| 19 new | it396 non-SCF | +0.1213 | +0.0272 | 7/19 | 0.1956 | [+0.0029, +0.2561] |
| 19 new | ωB97M-V (this work, SCF) | +0.1880 | +0.0451 | 5/19 | 0.0124 | [+0.0514, +0.3471] |

- GAS22 is significantly worse than it159, it396 and our ωB97M-V over all 35 sets: p 0.005, 0.023 and 0.003, and
  all CIs exclude zero.
- On the new sets it is significantly worse than it159 (p 0.011) and ωB97M-V (p 0.012). Against it396 the Wilcoxon
  test is not significant (p 0.20), although the bootstrap CI just excludes zero.

## 7. Caveats

- Same scope as jobs 0001/0002: 35 of 137 GSCDB sets, size-restricted, with partial ranking agreement with the full
  GSCDB (Spearman 0.93, Kendall 0.78).
- Non-SCF on ωB97M-V densities. For it159 the SCF/non-SCF difference was ≤ 0.004 NER. GAS22 differs more from
  ωB97M-V than it159 does (large heavy-atom shifts), so its density-driven error could be larger. A GAS22 SCF run
  (~45 GPU-h) would settle that, but a gap of 0.13 NER is unlikely to close.
- The reference states are those of job 0001 for all functionals.
- On MGCDB84, GAS22 had the best test WRMSD (3.600 vs 3.818 for it159). On this GSCDB subset the order reverses.
  A plausible reason is that GSCDB weights transition metals, heavy elements and properties, which MGCDB84 lacks.
  This is a description, not a basis for any selection.

## 8. Next steps (for the user)

- Ranking on this subset (mean NER, 35 sets): our ωB97M-V 1.086 < it159 1.099/1.101 < it396 1.110 < GAS22 1.228.
  it159 remains the best of our candidates. The it159–ωB97M-V gap is not significant.
- Optional: a GAS22 SCF run to rule out a density-driven effect (≈ 45–50 GPU-h). It is not needed to establish
  the order.
- The open items from job 0001 still apply: the AE11_Rn CPU-XC fallback and the extension to ≤ 1000 bf.

## Addendum (2026-10-24): AE11_Rn added, all 3,487 molecules

AE11_Rn could not be computed on the GPU (GPU4PySCF semilocal-XC defect for all-electron Rn). It was computed on the CPU for
GAS22 non-SCF with `runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py`, from the stored CPU reference orbitals (the reference is
within 0.003 kcal/mol of Q-Chem; the script was validated on AE11_Kr to 5e-6 kcal/mol). `molecules.jsonl`, `result.json` and
`energies.csv` now cover 3,487 molecules; the old files are `result_before_AE11_Rn.json` and `energies_before_AE11_Rn.csv`.
Everything above is the 35-set analysis and is unchanged. **36 sets (AE11 included):** GAS22 non-SCF 1.212 (SCF 1.190, from job 0004), rank 3 of 16. AE11 NER 0.67 (ωB97M-V 3.15, it159 2.53): the GAS22 gap to it159 shrinks to +0.071 (non-SCF) / +0.052 (SCF) and the bootstrap CI now includes zero. Note: the 35-set GAS22 non-SCF mean is 1.227, not 1.228 as written in §2.
Full 36-set tables and paired tests: `runs/job0004_GAS22_SCF_datasets/REPORT.md` §11.

## Addendum (2026-10-24): convergence fix

The two unconverged ωB97M-V references were converged on the CPU and replaced; GAS22 non-SCF of those two molecules recomputed (change ≤ 1e-5 kcal/mol). Mean NER unchanged. Details: `runs/convergence_fix/README.md` and `runs/job0004_GAS22_SCF_datasets/REPORT.md` §12. The numbers above are those before the fix, where they differ.
Before-fix files: `molecules_before_convergence_fix.jsonl`, `result_before_convergence_fix.json`.
