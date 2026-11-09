# GSCDB, 36 sets ≤ 500 basis functions: self-consistent evaluation of iteration 159

Job 0001, label it159. Written 2026-10-23. Protocol, dataset facts and the full fix log are in `GSCDB_NOTES.md`
(section "Job 0001"). Reporting only: nothing about it159 was changed or selected on the basis of these results.

## 1. Setup

| item | value |
|---|---|
| job | 1, unit `gscdb-job0001`, `runs/job0001_it159_datasets/` |
| dates | started 2026-10-20 23:27, finished 2026-10-23 01:56 |
| wall time | 48.1 h of unit time in 4 segments (8.4 + 1.9 + 7.2 + 30.6 h; stops were for bug fixes) |
| GPU time | 138.2 GPU-h of per-molecule compute: 111.0 in the final records, 20.1 in records invalidated by the fixes, 7.0 on AE11_Rn (excluded). Plus a few GPU-h of diagnostics. Estimate was 41–68; the 2× ceiling was 140. |
| molecules | 3,487 in the subset; 3,486 used; 1 excluded (AE11_Rn); 0 errors |
| unconverged | ωB97M-V reference: 2 (energies agree with Q-Chem to 0.0025 and 0.0003 kcal/mol). it159 SCF: 15 (see §7). |
| sets compared | 35 of 36. AE11 is incomplete because AE11_Rn is excluded. |

Protocol in three lines: ωB97M-V SCF in the GSCDB setup (basis/ECP/field/grids from the Q-Chem inputs; RI-JK only
for def2; state search against Q-Chem's ωB97M-V energy), orbitals stored; it159 non-SCF on that density; it159 SCF
from those orbitals. Metric: the GSCDB `analyze.ipynb` NER, reproduced to 2e-9 (and here exactly: the published
36-set means come out as ωB97M2 1.001, ωB97M-V 1.135, CF22D 1.231,
ωB97X-V 1.263, M06-2X 1.42).

**Code fixes made during the run (GSCDB-only files `dftsr/gscdb_scf.py`, `scripts/gscdb_scf.py`; MGCDB84 code,
CUDA and drivers untouched; originals in `code_backup_pre_fix/`).** Three GPU4PySCF 1.8.1 defects for h shells
(l = 5) were found; none was visible in the probe or the validation sets, which had no multi-atom h-shell molecules.
1. Nuclear-attraction integral wrong by up to 10 Eh on h elements → CPU core Hamiltonian for every l ≥ 5 molecule
   (it was only used with ECPs). Without it, aug-pc-4 diatomics collapsed 33,000–108,000 kcal/mol below Q-Chem.
2. Fused range-separated exchange (`get_k` with lr/sr factors) wrong for h shells (2.7 kcal/mol for He–Ar at 1.85 Å,
   10 kcal/mol for LiF at fixed density) → split into full-range and long-range K.
3. Semilocal XC matrix wrong for all-electron Rn (largest exponent 2.3e8) → not fixed; AE11_Rn excluded.
Plus: the VV10 SG-1 grid guard used ECP-reduced charges (4d atoms crashed); the CPU stability step received a
CuPy core Hamiltonian; the stability re-SCF had no fallbacks; and a field-sibling guess was added to the state
search for finite-field sets. All 467 h-shell molecules were recomputed with the final code.

## 2. Main table: mean NER over the 35 complete sets

it159 SCF ranks **3** among the 16 published functionals (behind ωB97M2 and ωB97M-V).

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.930 |
| 2 | ωB97M-V | 1.077 |
| 3 | ωB97M-V (this work, SCF) | 1.086 |
| 4 | **it159 SCF** | 1.099 |
| 5 | it159 non-SCF (ωB97M-V density) | 1.101 |
| 6 | ωB97X-V | 1.197 |
| 7 | CF22D | 1.233 |
| 8 | M062X | 1.422 |
| 9 | MN15 | 1.426 |
| 10 | revDSD-PBEP86-D4 | 1.526 |
| 11 | M052X | 1.577 |
| 12 | M08HX | 1.595 |
| 13 | CAMB3LYP | 1.871 |
| 14 | PBE0 | 1.949 |
| 15 | PW6B95 | 2.010 |
| 16 | r2SCAN0 | 2.143 |
| 17 | B3LYP | 2.463 |
| 18 | SOGGA11X | 2.693 |
| 19 | BMK | 3.352 |

For reference, the published means over all 36 sets: ωB97M2 1.001, ωB97M-V 1.135, CF22D 1.231, ωB97X-V 1.263,
M06-2X 1.42. AE11 is a hard set (published ωB97M-V NER 3.15), so dropping it lowers the means; compare within a
table, not across tables.

Sensitivity to the electronic-state differences (§7). it159 SCF and our ωB97M-V share the same states, so their
paired comparison is unaffected; only the comparison with the published functionals can shift.

| variant | sets | it159 SCF | rank among the 16 | ωB97M-V (ours) | ωB97M-V (published) |
|---|---|---|---|---|---|
| (a) 29 sets without state-affected molecules | 29 | 1.086 | 3 | 1.077 | 1.078 |
| (a) the same, new sets only | 16 | 1.230 | 4 | 1.221 | 1.221 |
| (b) it159 SCF + (published − our) ωB97M-V NER per set, 35 sets | 35 | 1.090 | 3 | 1.086 | 1.077 |
| (b) the same, 19 new sets | 19 | 1.197 | 4 | 1.191 | 1.186 |

On the 29 unaffected sets our ωB97M-V and the published ωB97M-V agree (1.077 vs 1.078). The rank of it159 does not
change in any variant.

## 3. New sets vs sets that overlap MGCDB84

**New relative to MGCDB84 (19 of the 20 sets; AE11 excluded).** This group carries the transferability claim.
it159 SCF ranks **4** among the 16 published functionals.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.953 |
| 2 | ωB97X-V | 1.155 |
| 3 | ωB97M-V | 1.186 |
| 4 | ωB97M-V (this work, SCF) | 1.191 |
| 5 | **it159 SCF** | 1.202 |
| 6 | it159 non-SCF (ωB97M-V density) | 1.204 |
| 7 | revDSD-PBEP86-D4 | 1.311 |
| 8 | MN15 | 1.339 |

**Updated versions of MGCDB84 sets (16 sets).** it159 SCF ranks **3**.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.903 |
| 2 | ωB97M-V | 0.947 |
| 3 | ωB97M-V (this work, SCF) | 0.961 |
| 4 | **it159 SCF** | 0.976 |
| 5 | it159 non-SCF (ωB97M-V density) | 0.979 |
| 6 | CF22D | 1.063 |
| 7 | M062X | 1.246 |
| 8 | ωB97X-V | 1.247 |

By MGCDB84 role (it159 was fitted on "train" and selected on "validation"):

| role | sets | rank of it159 SCF | top of the table |
|---|---|---|---|
| train | 5 | 2 | M062X 0.838, **it159 SCF** 0.981, it159 non-SCF (ωB97M-V density) 0.985, M08HX 1.005, ωB97M-V (this work, SCF) 1.013, ωB97M-V 1.013 |
| validation | 10 | 3 | ωB97M2 0.825, ωB97M-V 0.958, ωB97M-V (this work, SCF) 0.980, **it159 SCF** 1.004, it159 non-SCF (ωB97M-V density) 1.006, CF22D 1.040 |
| test | 1 | 3 | ωB97M-V 0.517, ωB97M-V (this work, SCF) 0.517, ωB97M2 0.676, **it159 SCF** 0.676, it159 non-SCF (ωB97M-V density) 0.682, revDSD-PBEP86-D4 0.819 |

The ordering is consistent with fitting: it159 beats ωB97M-V on the train sets (0.981 vs 1.013), is slightly
behind it on the validation sets (1.004 vs 0.980) and on the one test set (0.676 vs 0.517), and is slightly behind
it on the new sets (1.202 vs 1.191).

## 4. Mean NER per data type (35 sets)

| data type | sets | it159 SCF | it159 non-SCF | ωB97M-V (ours) | ωB97M-V (published) |
|---|---|---|---|---|---|
| Barrier Height | 2 | 1.038 | 1.038 | 1.057 | 0.972 |
| Electric field | 3 | 1.454 | 1.457 | 1.398 | 1.403 |
| Frequency | 1 | 1.116 | 1.117 | 0.926 | 0.925 |
| Isomerization | 1 | 1.322 | 1.326 | 1.166 | 1.166 |
| Noncovalent | 9 | 0.871 | 0.872 | 0.931 | 0.932 |
| Thermochemistry | 16 | 1.065 | 1.068 | 1.054 | 1.043 |
| Transition Metal | 3 | 1.569 | 1.571 | 1.459 | 1.462 |

it159 is better than ωB97M-V on noncovalent sets (0.871 vs 0.931), similar on thermochemistry and barrier heights,
and worse on transition metals (1.569 vs 1.459), electric-field properties (1.454 vs 1.398), the one frequency set
(V30: 1.116 vs 0.926) and the one isomerization set (1.322 vs 1.166).

## 5. Per-set table

Metric = the GSCDB metric (kcal/mol, or relative error for Pol130/OEEF, weighted for TMD10/O24/O24x4, relative
for Dip146, cm⁻¹ for V30). Winner among the 19 methods (16 published + our three).

| set | group | type | n | metric it159 SCF | metric it159 non-SCF | metric ωB97M-V (ours) | NER it159 SCF | NER it159 non-SCF | NER ωB97M-V (ours) | NER ωB97M-V (publ.) | ΔNER it159 − ours | winner (19 methods) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3d4dIPSS | new | Transition Metal | 32 | 5.6128 | 5.6159 | 5.0728 | 1.081 | 1.081 | 0.977 | 0.983 | +0.104 | revDSD-PBEP86-D4 |
| AE18 | train | Thermochemistry | 18 | 4.9229 | 4.9247 | 5.3086 | 0.994 | 0.995 | 1.072 | 1.072 | -0.078 | M062X |
| ALKBDE10 | new | Thermochemistry | 10 | 3.5611 | 3.5683 | 3.5811 | 1.094 | 1.096 | 1.100 | 1.100 | -0.006 | MN15 |
| BDE99MR | validation | Thermochemistry | 16 | 4.2218 | 4.2224 | 4.0702 | 1.274 | 1.274 | 1.228 | 1.117 | +0.046 | ωB97M2 |
| BDE99nonMR | train | Thermochemistry | 83 | 1.5378 | 1.5563 | 1.5260 | 0.796 | 0.806 | 0.790 | 0.790 | +0.006 | ωB97M2 |
| BH76RC | validation | Thermochemistry | 30 | 0.8993 | 0.9024 | 0.8518 | 0.894 | 0.897 | 0.846 | 0.847 | +0.048 | ωB97M2 |
| CT20 | validation | Noncovalent | 20 | 0.0687 | 0.0687 | 0.0731 | 0.769 | 0.768 | 0.818 | 0.817 | -0.049 | it159 non-SCF (ωB97M-V density) |
| DAPD | new | Transition Metal | 12 | 6.6251 | 6.6315 | 6.2636 | 2.386 | 2.388 | 2.256 | 2.256 | +0.130 | PW6B95 |
| DBH22 | train | Barrier Height | 22 | 1.1493 | 1.1511 | 1.1745 | 0.953 | 0.954 | 0.973 | 0.974 | -0.020 | ωB97M2 |
| DIPCS9 | new | Thermochemistry | 9 | 4.1885 | 4.1831 | 5.1060 | 1.238 | 1.237 | 1.510 | 1.510 | -0.272 | ωB97M2 |
| Dip146 | new | Electric field | 190 | 0.0391 | 0.0392 | 0.0365 | 1.125 | 1.128 | 1.051 | 1.050 | +0.074 | ωB97M2 |
| G21EA | train | Thermochemistry | 25 | 1.3359 | 1.3434 | 1.3691 | 0.976 | 0.982 | 1.001 | 1.001 | -0.025 | PW6B95 |
| HAT707MR | validation | Thermochemistry | 202 | 3.5752 | 3.5782 | 3.5028 | 1.113 | 1.114 | 1.090 | 1.031 | +0.023 | ωB97M2 |
| HAT707nonMR | validation | Thermochemistry | 505 | 1.8147 | 1.8178 | 1.9028 | 0.767 | 0.769 | 0.805 | 0.805 | -0.038 | ωB97M2 |
| HCP32 | new | Noncovalent | 32 | 0.5154 | 0.5144 | 0.5026 | 0.901 | 0.899 | 0.879 | 0.879 | +0.022 | ωB97X-V |
| HEAVY28 | new | Noncovalent | 28 | 0.1552 | 0.1554 | 0.1783 | 0.821 | 0.822 | 0.943 | 0.946 | -0.122 | revDSD-PBEP86-D4 |
| He3 | new | Noncovalent | 49 | 0.1715 | 0.1715 | 0.1830 | 0.973 | 0.973 | 1.038 | 1.038 | -0.065 | revDSD-PBEP86-D4 |
| IP23 | new | Thermochemistry | 23 | 2.4610 | 2.4599 | 2.5188 | 0.969 | 0.968 | 0.991 | 1.049 | -0.022 | ωB97M2 |
| ISOMERIZATION20 | validation | Isomerization | 20 | 1.7284 | 1.7345 | 1.5245 | 1.322 | 1.326 | 1.166 | 1.166 | +0.156 | ωB97M2 |
| MB08-165 | new | Thermochemistry | 165 | 3.9861 | 4.0085 | 3.7905 | 0.872 | 0.876 | 0.829 | 0.829 | +0.043 | ωB97M-V |
| NC11 | validation | Noncovalent | 11 | 0.0413 | 0.0413 | 0.0451 | 0.760 | 0.759 | 0.829 | 0.833 | -0.069 | revDSD-PBEP86-D4 |
| O24 | new | Noncovalent | 24 | 0.9027 | 0.9083 | 1.0280 | 0.803 | 0.808 | 0.914 | 0.912 | -0.111 | revDSD-PBEP86-D4 |
| O24x4 | new | Noncovalent | 96 | 0.5803 | 0.5830 | 0.6032 | 0.828 | 0.831 | 0.860 | 0.859 | -0.032 | ωB97M2 |
| OEEF | new | Electric field | 136 | 0.1029 | 0.1029 | 0.0993 | 1.175 | 1.174 | 1.134 | 1.134 | +0.041 | ωB97M2 |
| ORBH35 | new | Barrier Height | 35 | 2.5436 | 2.5432 | 2.5836 | 1.122 | 1.122 | 1.140 | 0.969 | -0.018 | CF22D |
| P34EA | new | Thermochemistry | 9 | 2.9973 | 3.0067 | 2.7914 | 1.909 | 1.915 | 1.777 | 1.771 | +0.132 | ωB97M2 |
| Pol130 | new | Electric field | 296 | 0.0490 | 0.0491 | 0.0478 | 2.062 | 2.068 | 2.010 | 2.023 | +0.052 | r2SCAN0 |
| RG10N | validation | Noncovalent | 275 | 0.0845 | 0.0847 | 0.0927 | 0.795 | 0.797 | 0.872 | 0.873 | -0.077 | ωB97M2 |
| SIE4x4 | new | Thermochemistry | 16 | 10.3684 | 10.3683 | 10.6122 | 1.130 | 1.130 | 1.156 | 1.156 | -0.026 | revDSD-PBEP86-D4 |
| SN13 | validation | Thermochemistry | 13 | 0.5906 | 0.5915 | 0.5046 | 0.936 | 0.937 | 0.799 | 0.799 | +0.137 | ωB97M-V |
| TA13 | train | Noncovalent | 13 | 1.6116 | 1.6131 | 1.6664 | 1.188 | 1.189 | 1.228 | 1.228 | -0.040 | revDSD-PBEP86-D4 |
| TAE_W4-17MR | validation | Thermochemistry | 17 | 4.3340 | 4.3500 | 4.1383 | 1.409 | 1.414 | 1.345 | 1.289 | +0.064 | PW6B95 |
| TMD10 | new | Transition Metal | 10 | 7.0032 | 7.0160 | 6.4582 | 1.241 | 1.243 | 1.144 | 1.146 | +0.097 | CF22D |
| V30 | new | Frequency | 275 | 33.5479 | 33.5727 | 27.8332 | 1.116 | 1.117 | 0.926 | 0.925 | +0.190 | revDSD-PBEP86-D4 |
| WCPT6 | test | Thermochemistry | 6 | 0.4484 | 0.4525 | 0.3428 | 0.676 | 0.682 | 0.517 | 0.517 | +0.159 | ωB97M-V |
| AE11 | new | Thermochemistry | 11 | — | — | — | — | — | — | 3.154 | — | not compared: AE11_Rn excluded |

## 6. Paired comparison: it159 SCF vs ωB97M-V

ΔNER = NER(it159 SCF) − NER(ωB97M-V) per set (column in §5). Wilcoxon signed-rank test over sets; bootstrap of the
mean ΔNER over sets (10,000 draws, seed 0).

| sets | vs | mean ΔNER | median ΔNER | it159 better | Wilcoxon p | bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| all 35 | ωB97M-V (this work, SCF) | +0.0128 | +0.0061 | 17/35 | 0.385 | [-0.0184, +0.0427] |
| all 35 | ωB97M-V | +0.0220 | +0.0226 | 16/35 | 0.158 | [-0.0116, +0.0546] |
| 19 new | ωB97M-V (this work, SCF) | +0.0110 | +0.0224 | 9/19 | 0.465 | [-0.0386, +0.0572] |
| 19 new | ωB97M-V | +0.0162 | +0.0388 | 8/19 | 0.374 | [-0.0360, +0.0647] |

No difference is significant. it159 and ωB97M-V are statistically indistinguishable on this subset; the point
estimate slightly favours ωB97M-V (+0.01 NER against our ωB97M-V, +0.02 against the published one).

## 7. SCF quality

**Density-driven error** |E_SCF − E_nonSCF| for it159 (3,486 molecules): mean 0.038, median
0.031, max 1.29 kcal/mol; 1 molecule above 1 kcal/mol. MGCDB84 gave mean 0.115,
max 1.29. Worst: W4-17_p4 1.293, MB08-165_145 0.556, AE11_Hg 0.339, AE11_Yb 0.325, AE11_Ba 0.218, MB08-165_157 0.200, AE11_Xe 0.200, Dip146_HBH2BH2- 0.199, Dip146_HBH2BH2+ 0.198, MB08-165_129 0.190. The SCF and non-SCF mean NER differ by
0.002–0.004 everywhere, so the self-consistency of it159 changes nothing at the set level.

**Unconverged it159 SCF: 15 molecules.** All ran DIIS (150 cycles) and level shift (300 cycles); the Newton
fallback is not implemented for the torch functional (`NotImplementedError`). Another 16 converged only with the
level shift. it159 also converges slowly on Ar-containing rare-gas dimers (19 cycles on average vs 8 for ωB97M-V).

| molecule | how | |E_SCF − E_nonSCF| (kcal/mol) |
|---|---|---|
| Pol130_P2H42- | newton_failed:NotImplementedError | 0.119 |
| Pol130_P2H42+ | newton_failed:NotImplementedError | 0.114 |
| 3d4dIPSS_Mo_GS | newton_failed:NotImplementedError | 0.101 |
| Pol130_SiH4 | newton_failed:NotImplementedError | 0.078 |
| Dip146_SiH3F2- | newton_failed:NotImplementedError | 0.070 |
| 3d4dIPSS_Co_ES+ | newton_failed:NotImplementedError | 0.067 |
| Pol130_CH3SH1+ | newton_failed:NotImplementedError | 0.060 |
| Pol130_PH2OH2+ | newton_failed:NotImplementedError | 0.052 |
| Dip146_CH3Cl2- | newton_failed:NotImplementedError | 0.044 |
| ALKBDE10_k | newton_failed:NotImplementedError | 0.041 |
| O24x5_h2o-na_1.2 | newton_failed:NotImplementedError | 0.032 |
| 3d4dIPSS_V_GS | newton_failed:NotImplementedError | 0.026 |
| Dip146_SO22- | newton_failed:NotImplementedError | 0.023 |
| Dip146_SF22- | newton_failed:NotImplementedError | 0.016 |
| Pol130_SF2 | newton_failed:NotImplementedError | 0.016 |

Effect on the property sets: the 11 Pol130/Dip146 entries that contain one of these molecules move by at most
0.003 (≤ 0.05 % relative) between it159 SCF and non-SCF, far below their errors (3–5 %).

**Reference SCF:** 3455 by DIIS, 12 with level shift, 19 with the Newton
solver; 2 not converged (Pol130_CH2PH2−, RG10N_KrKr_3p900), both within 0.003 kcal/mol of Q-Chem and consistent
with their neighbours.

**State searches: 42 triggered.** 30 ended within 0.01 kcal/mol of Q-Chem. The others:
ORBH35_CO3B_S04T +8.394, IP23_C2_n -3.373, W4-17_o3 +2.944, 3d4dIPSS_Fe_GS +0.422, 3d4dIPSS_Cu_ES+ +0.385, 3d4dIPSS_V_ES +0.310, 3d4dIPSS_Ti_GS+ +0.220, 3d4dIPSS_Y_GS+ +0.105 kcal/mol.

**Our ωB97M-V vs the published Q-Chem ωB97M-V:** n = 3486, median |Δ| 3.6e-04 kcal/mol, mean
0.0099; 10 molecules above 0.1 and 4 above 1 kcal/mol. Worst: 3d4dIPSS_Sc_ES -13.649, ORBH35_CO3B_S04T +8.394, IP23_C2_n -3.373, W4-17_o3 +2.944, 3d4dIPSS_Fe_GS +0.422, 3d4dIPSS_Cu_ES+ +0.385, 3d4dIPSS_V_ES +0.310, 3d4dIPSS_Ti_GS+ +0.220, 3d4dIPSS_Pd_ES -0.113, 3d4dIPSS_Y_GS+ +0.105, OEEF_meoh_-0.05 +0.084, 3d4dIPSS_Nb_GS+ +0.073, 3d4dIPSS_V_GS+ +0.064, HEAVY28_pbh4_2 -0.062, P34_P63A_Se +0.055.
All molecules above 0.1 kcal/mol are electronic-state differences, not numerical ones:
- 3d4dIPSS_Sc_ES (quartet Sc, −13.6): we converge to a quartet below Q-Chem's.
- ORBH35_CO3B_S04T (singlet CO₃, +8.4) and W4-17_o3 (ozone, +2.9): Q-Chem found a lower state that neither the
  guesses nor stability following reach. W4-17_o3 enters BDE99MR, HAT707MR and TAE_W4-17MR.
- IP23_C2_n (singlet C₂, −3.4): the broken-symmetry guess finds a solution below Q-Chem's.
- Open-shell 3d/4d atoms (3d4dIPSS Fe, Cu⁺ ES, V ES, Ti⁺, Pd ES, Y⁺; 0.1–0.42): near-degenerate d occupations.
These five sets plus 3d4dIPSS are the "state-affected" sets of the sensitivity check in §2.

## 8. Caveats

- This is a size-restricted subset (all GSCDB sets whose largest molecule has ≤ 500 basis functions), and AE11 is
  missing. The claim is scoped to these 35 sets, not to GSCDB137.
- With the 16 published functionals, this subset's ranking agrees with the full GSCDB ranking only partly
  (Spearman 0.93, Kendall 0.78). Among pairs of functionals within 10 % of each other, the subset picked the other
  winner in 4 of 10 cases. it159 vs ωB97M-V is such a close pair (ΔNER ≈ 0.01).
- The subset leans towards small-molecule thermochemistry and property sets. It has almost no barrier heights
  (2 sets), no conformers and no larger noncovalent complexes.
- Q-Chem vs PySCF: on the 29 sets without state differences our ωB97M-V reproduces the published mean NER to
  0.001 (1.077 vs 1.078); molecule-level agreement has median 3.6e-4 kcal/mol. The remaining differences are
  electronic states for a handful of molecules (§7); they shift our ωB97M-V on 5 sets but do not change any
  conclusion (§2, sensitivity).
- The run needed six code changes mid-way (§1). All affected molecules were recomputed with the final code, and the
  ωB97M-V results were checked against Q-Chem molecule by molecule. The three GPU4PySCF defects should be reported
  upstream; until then any GSCDB molecule with h functions depends on the workarounds.

## 9. Next steps (for the user to decide)

- **GAS22 on the same subset.** It is not an MCP iteration: run `scripts/gscdb_scf.py` with `baselines/gas22.py`
  and `/home/ash/Documents/dft_functional_sr/runs/scf_gas22/params_published.json` (label GAS22), reusing the
  stored references. `gscdb_estimate` (now calibrated on this job): 50.4 GPU-h, ≈ 17 h on 3 GPUs. The measured
  candidate part of this job was 50.9 GPU-h, so ≈ 50 GPU-h is realistic (the reference SCFs are already stored).
  Not launched; needs approval.
- **AE11.** Including it needs a CPU fallback for the semilocal XC of AE11_Rn (reference and candidate). Small
  code change for one molecule; needs approval.
- **Extension** to the sets whose largest molecule has ≤ 1000 functions (pair-winner flip rate 1/10 for close pairs)
  or ≤ 2000. All stored references with h functions are now correct; molecules > 500 functions are new work.

## Addendum (2026-10-24): AE11_Rn added, all 3,487 molecules

AE11_Rn could not be computed on the GPU (GPU4PySCF semilocal-XC defect for all-electron Rn). It was computed on the CPU for
it159 SCF and non-SCF with `runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py`, from the stored CPU reference orbitals (the reference is
within 0.003 kcal/mol of Q-Chem; the script was validated on AE11_Kr to 5e-6 kcal/mol). `molecules.jsonl`, `result.json` and
`energies.csv` now cover 3,487 molecules; the old files are `result_before_AE11_Rn.json` and `energies_before_AE11_Rn.csv`.
Everything above is the 35-set analysis and is unchanged. **36 sets (AE11 included):** it159 SCF 1.139 / non-SCF 1.141 (rank 3 of the 16 published; rank 2 on the 20 new sets, 1.269). Our ωB97M-V 1.144, it159 − ωB97M-V −0.005 (p 0.58): tied. AE11 NER: it159 2.53, ωB97M-V 3.15. AE11_Rn density-driven error 0.28 kcal/mol (3rd–5th largest).
Full 36-set tables and paired tests: `runs/job0004_GAS22_SCF_datasets/REPORT.md` §11.

## Addendum (2026-10-24): convergence fix

it159 candidate SCF unconverged 15 → 0 (3 strict, 12 noise-limited) and the two unconverged ωB97M-V references converged on the CPU. Energy changes ≤ 1.8e-4 kcal/mol; mean NER unchanged (it159 SCF 36 sets 1.13872). Details: `runs/convergence_fix/README.md` and `runs/job0004_GAS22_SCF_datasets/REPORT.md` §12. The numbers above are those before the fix, where they differ.
Before-fix files: `molecules_before_convergence_fix.jsonl`, `result_before_convergence_fix.json`.
