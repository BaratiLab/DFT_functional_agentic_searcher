# GSCDB, 36 sets ≤ 500 basis functions: self-consistent evaluation of GAS22

Job 0004, label GAS22 (published parameters). Written 2026-10-24. Prompt: `GAS22_GSCDB_LE500_SCF_PROMPT.md` (root). Baselines:
`../job0001_it159_datasets/REPORT.md` (it159 SCF), `../job0002_it396_datasets/REPORT.md`,
`../job0003_GAS22_datasets/REPORT.md` (GAS22 non-SCF). Protocol and fix log: `GSCDB_NOTES.md`. Reporting only: nothing
about GAS22, it159 or any selection was changed on the basis of these results.

> **Update 2026-10-24 (convergence fix, §12).** The unconverged records described in §1, §7 and §8 were reworked: GAS22 candidate SCF 66 → 2
> unconverged (the same Pd atom, twice), it159 15 → 0, and the 2 unconverged ωB97M-V references → 0. §2–§11 numbers are unchanged to 1e-5. The counts below
> that say 66/15 describe the state before the fix.

## 1. Setup

| item | value |
|---|---|
| job | 4 (registry iteration_id 0), unit `gscdb-job0004`, `runs/job0004_GAS22_SCF_datasets/` |
| dates | created 2026-10-23 15:37; stopped 15:57 at the sanity gate (see below); resumed 16:42 with the user's go-ahead; finished 2026-10-24 14:13 |
| wall time | 21.8 h of unit time in 2 segments (0.3 + 21.5 h) |
| GPU time | 65.3 GPU-h of per-molecule time (estimate 50.4, 2× ceiling 100). The candidate part of it159's job was 50.9 GPU-h. The slowest 1 % of molecules (34) cost 22.5 GPU-h of this. |
| molecules | 3,487 in the subset. The runner did 3,486 and had AE11_Rn pre-excluded (error record). **AE11_Rn was computed afterwards with a CPU fallback, so all 3,487 are now done** (§10). All tables in §2–§9 are over the 35 sets that every method has, so they do not include AE11. |
| unconverged | candidate SCF: 66 (1.9 %), see §7. Reference: 2, both inherited from job 0001 (loaded orbitals). |
| sets compared | 35 of 36 (AE11 incomplete) |
| mode | SCF from the stored ωB97M-V orbitals (candidate non-SCF + SCF, as for it159), RI-JK for def2 and exact-integral final energy, the same code and settings as jobs 0001–0003 |

**Sanity gate (after 300 molecules).**
- Errors: only AE11_Rn. Pass.
- References: all records have `ref.loaded = true`, none `ref_redone`; `E_ref` equals job 0001's exactly (max difference 0.0). Pass.
- Non-SCF: `E_nonscf` equals job 0003's to 1.6e-9 kcal/mol over all 3,486 molecules (limit 1e-6). The setup is identical. Pass.
- Density-driven error: mean 0.141, max 1.69 kcal/mol, 2 of 303 above 1. Pass.
- **Unconverged candidate SCF: 20 of 303 (6.6 %), limit 1 %. Fail.** The job was stopped as the prompt required, and the
  failure reported. The user said to continue and deal with the unconverged ones later. The final rate is
  66/3,486 = 1.9 %, still above 1 %.

Not a fault of the setup: GAS22's SCF has trouble converging on light open-shell atoms and some other small molecules
(§7). The code is unchanged.

## 2. Main table: mean NER over the 35 complete sets

GAS22 SCF is **1.205** and ranks **4** among the 16 published functionals (behind ωB97M2, ωB97M-V and
ωB97X-V; ahead of CF22D). The SCF is 0.022 better than GAS22 non-SCF (1.227). It is still clearly behind it159 SCF
(1.099) and our ωB97M-V (1.086).

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.930 |
| 2 | ωB97M-V | 1.077 |
| 3 | ωB97M-V (this work) | 1.086 |
| 4 | it159 SCF | 1.099 |
| 5 | it159 non-SCF | 1.101 |
| 6 | it396 non-SCF | 1.110 |
| 7 | ωB97X-V | 1.197 |
| 8 | **GAS22 SCF** | 1.205 |
| 9 | GAS22 non-SCF | 1.227 |
| 10 | CF22D | 1.233 |
| 11 | M06-2X | 1.422 |
| 12 | MN15 | 1.426 |

Next published functionals: revDSD-PBEP86-D4 1.526, M05-2X 1.577, M08-HX 1.595, …

Reference values reproduced: ωB97M2 0.930, ωB97M-V 1.077, ωB97X-V 1.197, CF22D 1.233, ωB97M-V (ours) 1.086, it159
SCF 1.099 / non-SCF 1.101, it396 non-SCF 1.110. Job 0003's report gives GAS22 non-SCF as 1.228. The exact value is
1.22745, so it should read 1.227 (same energies, rounding slip).

Order on this subset: ωB97M-V (ours) 1.086 < it159 1.099 < it396 1.110 < **GAS22 SCF 1.205**.

## 3. The 19 new and 16 overlapping sets

**19 new sets (not in MGCDB84).** GAS22 SCF 1.353, rank **6**. For comparison it159 SCF 1.202,
it396 1.258, ωB97M-V (ours) 1.191, GAS22 non-SCF 1.379.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 0.953 |
| 2 | ωB97X-V | 1.155 |
| 3 | ωB97M-V | 1.186 |
| 4 | ωB97M-V (this work) | 1.191 |
| 5 | it159 SCF | 1.202 |
| 6 | it159 non-SCF | 1.204 |
| 7 | it396 non-SCF | 1.258 |
| 8 | revDSD-PBEP86-D4 | 1.311 |
| 9 | MN15 | 1.339 |
| 10 | **GAS22 SCF** | 1.353 |

**16 MGCDB84-derived sets.** GAS22 SCF 1.029, rank **3**. it159 SCF 0.976, it396
0.935, ωB97M-V (ours) 0.961, GAS22 non-SCF 1.047.

By MGCDB84 role:

| role | sets | GAS22 SCF | GAS22 non-SCF | it159 SCF | it396 non-SCF | ωB97M-V (ours) | rank of GAS22 SCF among 16 |
|---|---|---|---|---|---|---|---|
| train | 5 | 1.003 | 1.034 | 0.981 | 0.913 | 1.013 | 2 |
| validation | 10 | 1.087 | 1.100 | 1.004 | 0.960 | 0.980 | 4 |
| test | 1 | 0.579 | 0.584 | 0.676 | 0.794 | 0.517 | 2 |

On the single MGCDB84 test-role set (WCPT6) GAS22 SCF (0.579) beats it159 (0.676) and it396 (0.794). It is behind
ωB97M-V (0.517). On the five train and ten validation sets it is behind it159 and it396.

## 4. Mean NER per data type (35 sets)

| data type | sets | GAS22 SCF | GAS22 non-SCF | it159 SCF | it396 non-SCF | ωB97M-V (ours) | ωB97M-V (published) |
|---|---|---|---|---|---|---|---|
| Barrier Height | 2 | 1.072 | 1.087 | 1.038 | 1.022 | 1.057 | 0.972 |
| Electric field | 3 | 1.567 | 1.572 | 1.454 | 1.472 | 1.398 | 1.403 |
| Frequency | 1 | 1.066 | 1.068 | 1.116 | 1.139 | 0.926 | 0.925 |
| Isomerization | 1 | 1.421 | 1.433 | 1.322 | 1.160 | 1.166 | 1.166 |
| Noncovalent | 9 | 0.844 | 0.861 | 0.871 | 0.891 | 0.931 | 0.932 |
| Thermochemistry | 16 | 1.172 | 1.200 | 1.065 | 1.044 | 1.054 | 1.043 |
| Transition Metal | 3 | 2.169 | 2.204 | 1.569 | 1.790 | 1.459 | 1.462 |

GAS22 is best of the four only on noncovalent sets (0.844) and ahead of it159 on the one frequency set (V30). It is
clearly worse on transition metals (2.169 vs 1.569; 3d4dIPSS +0.95, TMD10 +0.66 NER against it159), on thermochemistry
(1.172 vs 1.065; SN13 +0.51, ALKBDE10 +0.31, MB08-165 +0.28) and on electric-field properties (1.567; OEEF +0.29).

## 5. Per-set table

Metric = the GSCDB metric (kcal/mol; relative error for Pol130/OEEF/Dip146; weighted for TMD10/O24/O24x4; cm⁻¹ for
V30). Winner among the 22 methods (16 published + ours + it159 SCF/non-SCF + it396 + GAS22 SCF/non-SCF).

| set | group | type | n | metric GAS22 SCF | metric GAS22 non-SCF | NER GAS22 SCF | NER GAS22 non-SCF | NER it159 SCF | ΔNER GAS22 SCF − it159 SCF | winner (22 methods) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3d4dIPSS | new | Transition Metal | 32 | 10.5200 | 10.8218 | 2.026 | 2.084 | 1.081 | +0.945 | revDSD-PBEP86-D4 |
| AE18 | train | Thermochemistry | 18 | 3.2829 | 3.3111 | 0.663 | 0.669 | 0.994 | -0.331 | M06-2X |
| ALKBDE10 | new | Thermochemistry | 10 | 4.5642 | 4.8379 | 1.402 | 1.486 | 1.094 | +0.308 | MN15 |
| BDE99MR | validation | Thermochemistry | 16 | 4.3761 | 4.3113 | 1.320 | 1.301 | 1.274 | +0.047 | ωB97M2 |
| BDE99nonMR | train | Thermochemistry | 83 | 1.7127 | 1.7789 | 0.887 | 0.921 | 0.796 | +0.091 | ωB97M2 |
| BH76RC | validation | Thermochemistry | 30 | 1.1257 | 1.1643 | 1.119 | 1.157 | 0.894 | +0.225 | ωB97M2 |
| CT20 | validation | Noncovalent | 20 | 0.0765 | 0.0765 | 0.856 | 0.856 | 0.769 | +0.087 | it159 non-SCF |
| DAPD | new | Transition Metal | 12 | 7.1592 | 7.0912 | 2.578 | 2.554 | 2.386 | +0.192 | PW6B95 |
| DBH22 | train | Barrier Height | 22 | 1.2276 | 1.2540 | 1.017 | 1.039 | 0.953 | +0.065 | ωB97M2 |
| DIPCS9 | new | Thermochemistry | 9 | 4.2154 | 4.1815 | 1.246 | 1.236 | 1.238 | +0.008 | ωB97M2 |
| Dip146 | new | Electric field | 190 | 0.0376 | 0.0376 | 1.080 | 1.080 | 1.125 | -0.044 | ωB97M2 |
| G21EA | train | Thermochemistry | 25 | 1.6462 | 1.7282 | 1.203 | 1.263 | 0.976 | +0.227 | it396 non-SCF |
| HAT707MR | validation | Thermochemistry | 202 | 3.6255 | 3.6813 | 1.128 | 1.146 | 1.113 | +0.016 | ωB97M2 |
| HAT707nonMR | validation | Thermochemistry | 505 | 1.9684 | 1.9654 | 0.832 | 0.831 | 0.767 | +0.065 | ωB97M2 |
| HCP32 | new | Noncovalent | 32 | 0.5031 | 0.5019 | 0.880 | 0.878 | 0.901 | -0.022 | ωB97X-V |
| HEAVY28 | new | Noncovalent | 28 | 0.1308 | 0.1314 | 0.692 | 0.695 | 0.821 | -0.129 | revDSD-PBEP86-D4 |
| He3 | new | Noncovalent | 49 | 0.1680 | 0.1680 | 0.953 | 0.953 | 0.973 | -0.020 | revDSD-PBEP86-D4 |
| IP23 | new | Thermochemistry | 23 | 2.6216 | 2.6333 | 1.032 | 1.037 | 0.969 | +0.063 | ωB97M2 |
| ISOMERIZATION20 | validation | Isomerization | 20 | 1.8585 | 1.8741 | 1.421 | 1.433 | 1.322 | +0.099 | ωB97M2 |
| MB08-165 | new | Thermochemistry | 165 | 5.2566 | 5.3265 | 1.149 | 1.165 | 0.872 | +0.278 | ωB97M-V |
| NC11 | validation | Noncovalent | 11 | 0.0342 | 0.0345 | 0.629 | 0.635 | 0.760 | -0.130 | revDSD-PBEP86-D4 |
| O24 | new | Noncovalent | 24 | 0.9755 | 1.0408 | 0.867 | 0.925 | 0.803 | +0.065 | revDSD-PBEP86-D4 |
| O24x4 | new | Noncovalent | 96 | 0.5933 | 0.6323 | 0.846 | 0.902 | 0.828 | +0.019 | ωB97M2 |
| OEEF | new | Electric field | 136 | 0.1285 | 0.1293 | 1.467 | 1.476 | 1.175 | +0.292 | ωB97M2 |
| ORBH35 | new | Barrier Height | 35 | 2.5542 | 2.5731 | 1.127 | 1.136 | 1.122 | +0.005 | CF22D |
| P34EA | new | Thermochemistry | 9 | 3.2641 | 3.5060 | 2.078 | 2.232 | 1.909 | +0.170 | ωB97M2 |
| Pol130 | new | Electric field | 296 | 0.0512 | 0.0514 | 2.155 | 2.161 | 2.062 | +0.093 | r2SCAN0 |
| RG10N | validation | Noncovalent | 275 | 0.0666 | 0.0669 | 0.626 | 0.630 | 0.795 | -0.169 | ωB97M2 |
| SIE4x4 | new | Thermochemistry | 16 | 10.6926 | 10.7035 | 1.165 | 1.166 | 1.130 | +0.035 | revDSD-PBEP86-D4 |
| SN13 | validation | Thermochemistry | 13 | 0.9138 | 0.9838 | 1.448 | 1.559 | 0.936 | +0.512 | ωB97M-V |
| TA13 | train | Noncovalent | 13 | 1.6902 | 1.7338 | 1.246 | 1.278 | 1.188 | +0.058 | revDSD-PBEP86-D4 |
| TAE_W4-17MR | validation | Thermochemistry | 17 | 4.5908 | 4.4617 | 1.492 | 1.450 | 1.409 | +0.083 | PW6B95 |
| TMD10 | new | Transition Metal | 10 | 10.7338 | 11.1399 | 1.902 | 1.974 | 1.241 | +0.661 | CF22D |
| V30 | new | Frequency | 275 | 32.0211 | 32.0930 | 1.066 | 1.068 | 1.116 | -0.051 | revDSD-PBEP86-D4 |
| WCPT6 | test | Thermochemistry | 6 | 0.3837 | 0.3875 | 0.579 | 0.584 | 0.676 | -0.098 | ωB97M-V |
| AE11 | new | Thermochemistry | 11 | — | — | — | — | — | — | not compared: AE11_Rn excluded |

GAS22 SCF is the winner on no set. SCF versus non-SCF changes the NER of a set by up to 0.154
(P34EA -0.154, SN13 -0.111, ALKBDE10 -0.084); it159's SCF changed
nothing at set level (≤ 0.004).

## 6. Paired comparisons

ΔNER = NER(GAS22 SCF) − NER(other) per set; positive = GAS22 worse. Wilcoxon signed-rank test over sets; bootstrap of
the mean ΔNER over sets (10,000 draws, `numpy.random.default_rng(0)`).

| sets | vs | mean ΔNER | median ΔNER | GAS22 SCF better | Wilcoxon p | bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| all 35 | it159 SCF | +0.1061 | +0.0647 | 9/35 | 0.0060 | [+0.0347, +0.1899] |
| all 35 | ωB97M-V (this work) | +0.1189 | +0.0440 | 9/35 | 0.0078 | [+0.0296, +0.2157] |
| all 35 | GAS22 non-SCF | -0.0223 | -0.0064 | 27/35 | 0.0003 | [-0.0358, -0.0103] |
| 19 new | it159 SCF | +0.1509 | +0.0632 | 5/19 | 0.0160 | [+0.0458, +0.2772] |
| 19 new | ωB97M-V (this work) | +0.1619 | +0.0405 | 6/19 | 0.0446 | [+0.0315, +0.3135] |
| 19 new | GAS22 non-SCF | -0.0261 | -0.0064 | 14/19 | 0.0082 | [-0.0468, -0.0086] |

- **GAS22 SCF vs it159 SCF:** significantly worse over all 35 sets (+0.106, p 0.006, CI excludes zero) and over the 19
  new sets (+0.151, p 0.016). Same verdict as the non-SCF comparison (+0.126, p 0.005).
- **GAS22 SCF vs ωB97M-V (ours):** significantly worse (+0.119, p 0.008; new sets +0.162, p 0.045).
- **GAS22 SCF vs non-SCF:** the density-driven effect on the metric is −0.022 NER (better by SCF), 27 of 35 sets
  improve, p 0.0003. It is real but small, about five times smaller than the gap to it159 (+0.106). It does not change any
  conclusion of job 0003. (For comparison: it396 non-SCF vs GAS22 SCF +0.095, p 0.058 over 35 sets, not significant
  at 5 %; on the new sets +0.095, p 0.42.)

## 7. SCF quality

**Density-driven error** |E_SCF − E_nonSCF| (3,486 molecules), GAS22 against it159:

| | mean | median | max | > 1 kcal/mol | > 0.3 | > 0.1 |
|---|---|---|---|---|---|---|
| GAS22 | 0.150 | 0.121 | 1.915 | 10 | 246 | 2189 |
| it159 | 0.038 | 0.031 | 1.293 | 1 | 4 | 129 |

The flag levels (mean > 0.3, or more than about 1 % of molecules above 1) are not reached (0.29 % above 1). GAS22's
density-driven error is about four times it159's. Worst: TMD17_CrS 1.91, TMD18_MnBr 1.91, W4-17_p4 1.76, ALKBDE10_mgo 1.69, TMD15_TiBr 1.43, P34_P66A_Sn 1.29, G21EA_EA_si- 1.17, TMD19_FeCl 1.07, G21EA_EA_c- 1.05, P34_P61A_Ge 1.04 kcal/mol. By set it is largest
for TMD10 0.51, P34EA 0.50, 3d4dIPSS 0.47, BDE99nonMR 0.29, G21EA 0.27 (mean, kcal/mol; mostly transition-metal sets and atoms), with it159 at 0.03–0.17 on the same sets.

**Unconverged candidate SCF: 66 molecules (1.9 %).** it159 had 15 (5 of them also unconverged here). All 66
ran 150 DIIS and 300 level-shift cycles, then the Newton fallback raised `NotImplementedError` (not implemented for
torch functionals). Another 15 molecules converged only with the level shift (it159: 16); 3405 converged by DIIS.
By set: O24x4 17, Pol130 12, 3d4dIPSS 10, Dip146 6, ALKBDE10 3, TMD10 2, O24 2, RG10N 2, BH76RC 2, DBH22 2, TA13 2, G21EA 1, AE18 1, ISOMERIZATION20 1, HEAVY28 1, NC11 1, DAPD 1. Typical cases are light open-shell atoms (Li, Na, K, Be⁺) and
every molecule set that reuses them, the 3d/4d atoms of 3d4dIPSS, HCl, HCN, SO₂, PH₃, P₂H₄ and their field points.
All of them sit within 0.84 kcal/mol of the non-SCF energy. The largest:

| molecule | |E_SCF − E_nonSCF| GAS22 | same molecule, it159 | it159 converged |
|---|---|---|---|
| TMD20_FeS | 0.837 | 0.096 | True |
| 3d4dIPSS_Co_ES+ | 0.808 | 0.067 | False |
| 3d4dIPSS_Pd_ES | 0.754 | 0.075 | True |
| 3d4dIPSS_Co_GS | 0.750 | 0.068 | True |
| 3d4dIPSS_Fe_GS+ | 0.690 | 0.058 | True |
| 3d4dIPSS_Rh_GS | 0.677 | 0.071 | True |
| 3d4dIPSS_Cu_ES | 0.666 | 0.090 | True |
| 3d4dIPSS_Mn_ES | 0.655 | 0.080 | True |
| ALKBDE10_k | 0.531 | 0.041 | False |
| TMD09_Cr | 0.425 | 0.042 | True |
| Pol130_H2O-Li0- | 0.411 | 0.038 | True |
| Pol130_H2O-Li1+ | 0.410 | 0.038 | True |

Cause not investigated: GAS22's potential is probably rougher than it159's for very diffuse, low-density atoms, so
DIIS does not settle. No code was changed. Property sets: the Pol130/Dip146 entries that contain an unconverged
molecule still have NER equal to non-SCF within 0.01 (Pol130 2.155 vs 2.161, Dip146 1.080 vs 1.080).

**Slowest molecules.** The 34 slowest (1 %) cost 22.5 GPU-h of 65.3.

| molecule | time | how |
|---|---|---|
| RG10N_KrKr_3p900 | 221 min | newton_failed:NotImplementedError |
| RG10N_KrKr_4p100 | 210 min | newton_failed:NotImplementedError |
| RG10N_KrKr_2p200 | 93 min | level_shift |
| RG10N_KrKr_2p400 | 91 min | level_shift |
| RG10N_NeKr_4p800 | 52 min | level_shift |
| RG10N_NeKr_5p000 | 49 min | level_shift |
| P34_P67A_Sb | 41 min | level_shift |
| Pol130_P2H4 | 38 min | newton_failed:NotImplementedError |
| 21_Na_AE18 | 37 min | newton_failed:NotImplementedError |
| Pol130_P2H42+ | 37 min | newton_failed:NotImplementedError |

The two Kr–Kr dimers at 3.9 and 4.1 Å took 3.5–3.7 h each and ended unconverged. Kr–Kr at 2.2/2.4 Å and Ne–Kr took 50–95 min
with the level shift. Then come Pol130 field points of P₂H₄, CH₃Cl, H₂O–Li and SO₂ at 25–40 min. This is the
fallback-solver tail the prompt described. GPU utilisation was 100 % on two cards during the tail, so it was computing, not hanging.

**Reference side.** All 3,486 references were loaded (none computed or redone); `E_ref` is identical to job 0001's, so
our ωB97M-V agreement with Q-Chem and the state differences are those of job 0001 (median 3.6e-4 kcal/mol; 4 molecules
above 1 kcal/mol, all electronic-state differences).

## 8. Caveats

- Scope: 35 of 137 GSCDB sets, size-restricted. With the 16 published functionals the subset ranking agrees with the full
  GSCDB ranking only partly (Spearman 0.93, Kendall 0.78).
- AE11 is excluded (AE11_Rn cannot be computed on the GPU), so all means are over 35 sets.
- The reference states are those of job 0001 for every functional. SCF finds its own state from those orbitals; for
  GAS22 the state-affected molecules (Sc_ES, CO₃ triplet, O₃, C₂⁻) could in principle relax differently. No check was done.
- GAS22 fails the UEG-exchange constraint (g_x = 0.862, `project_details.md` §3). This matches its large systematic
  heavy-atom total-energy shifts against it159 (Hg −888, Kr −236 kcal/mol) and its transition-metal result.
- The MGCDB84 test order (GAS22 best, 3.600 vs it159 3.818) is reversed on this subset: it159 is better by 0.106 NER
  (p 0.006). WCPT6 is the one set that keeps the MGCDB84 order. A description, not a basis for any selection.
- The unconverged-SCF rate (1.9 %) exceeds the 1 % gate. Energies of those molecules are not at a variational minimum
  (|ΔE| ≤ 0.84 kcal/mol from non-SCF), so SCF numbers for them are slightly uncertain. The effect on set NER
  is not separable from the finite-difference property sets and was not tested.

## 9. Next steps (for the user to decide; nothing done without approval)

- **Convergence of torch functionals.** A second-order or damped fallback would remove the 66 unconverged records and
  most of the 23 GPU-h tail of the slowest 1 %. Code change; applies to any later SCF run, also it159's 15.
- **AE11_Rn** CPU fallback for the semilocal XC (brings AE11 back as a 36th set).
- **Extension** to the sets whose largest molecule has ≤ 1000 functions. Reference orbitals for those are not stored.
  Estimate first.
- **No further GAS22 or it396 SCF run is needed**: the order ωB97M-V (ours) < it159 < it396 < GAS22 holds with SCF
  too. it159 remains the best of our candidates and is statistically indistinguishable from ωB97M-V.
- Upstream report of the three GPU4PySCF h-shell defects is still open.

## 10. Addendum (2026-10-24): AE11_Rn computed, job complete at 3,487 molecules

The user asked for all 3,487 molecules. AE11_Rn was the only one missing: GPU4PySCF's semilocal XC is wrong for all-electron Rn
(largest exponent 2.3e8). It was computed on the CPU instead, with a standalone script that changes no project code:
`ae11_cpu_fallback.py` in this folder.

**Method.** The whole SCF runs on the CPU with PySCF: exact in-core J/K for the full-range and ω = 0.3 ERIs, libxc ωB97M-V for the
reference, the candidate's PyTorch semilocal XC for GAS22, VV10 on the CPU grids. The grids (250, 974 XC; SG-1 → (50, 194)
VV10), thresholds and SCF fallbacks are those of `dftsr/gscdb_scf.py`. Same protocol: ωB97M-V SCF, GAS22 non-SCF on that
density, GAS22 SCF from those orbitals.

**Validation on AE11_Kr** (exponent 1.8e8, which the GPU does correctly): CPU − GPU = −5.2e-6 kcal/mol (ωB97M-V E_ref),
−4.9e-6 (GAS22 non-SCF), −5.3e-6 (GAS22 SCF); vs the published Q-Chem ωB97M-V 7.6e-5 kcal/mol. 35 s.

**AE11_Rn result** (61 s): ωB97M-V − published Q-Chem = +0.0032 kcal/mol (the same size as the other heavy atoms, e.g. Hg
−0.003). All SCFs converged by DIIS (10 and 7 cycles). GAS22 non-SCF − ωB97M-V = −1221 kcal/mol (heavy-atom shift, §1 of
job 0003); density-driven error −0.33 kcal/mol.

**Files.**
- `molecules.jsonl` now has 3,487 records; the Rn record has `cpu_fallback: true`, `ref.loaded: false` and `gpu: -1`. It is
  the one exception to the gate rule "all references loaded".
- The old pre-exclusion record is `molecules_excluded.jsonl.pre_cpu_fallback`.
- The stored reference `AE11_Rn.h5` was **replaced** by the CPU orbitals (the old one was divergent, E ≈ −2.8e13 Eh); the old
  file is kept as `AE11_Rn.h5.before_cpu_fallback` here. A GPU run on AE11_Rn would still give wrong results, so the
  other jobs' pre-exclusion of it stays correct for the GPU path.
- `result.json` and `energies.csv` were regenerated over 3,487 molecules; `progress.json` now shows `n_done` 3,487 and
  `n_errors` 0 (with a note). The runner's originals are `*_runner_with_AE11_Rn.*`.

**What AE11 changes for GAS22.** AE11 is a set of absolute total energies of heavy atoms. Over all 36 sets, using only the
methods that now have AE11 (our ωB97M-V, GAS22 SCF/non-SCF and the 16 published):

| method | AE11 metric (kcal/mol) | AE11 NER | mean NER, 36 sets |
|---|---|---|---|
| ωB97M2 (published) | 410.1 | 3.51 | 1.001 |
| ωB97M-V (published) | 368.9 | 3.15 | 1.135 |
| ωB97M-V (this work) | 368.9 | 3.15 | 1.144 |
| **GAS22 SCF** | **78.4** | **0.67** | **1.190** (rank 3 of 16) |
| GAS22 non-SCF | 78.3 | 0.67 | 1.212 |
| CF22D (published) | 137.5 | 1.18 | 1.231 |

- GAS22 is by far the best of these on AE11 (NER 0.67 against 3.15 for ωB97M-V). Its large systematic heavy-atom
  total-energy shift, which cancels in reactions, brings the absolute energies much closer to the reference. So the 35-set
  tables above understate GAS22 on this one set.

The 36-set comparison with it159 and it396 is in §11 (their AE11_Rn was computed afterwards, same CPU path).

## 11. All 36 sets: it159, it396, GAS22 and ωB97M-V on the complete subset (2026-10-24)

AE11_Rn was computed with the CPU fallback for every job (job 0001 it159 SCF + non-SCF, job 0002 it396 non-SCF, job 0003
GAS22 non-SCF, job 0004 GAS22 SCF), using the stored CPU reference orbitals. Each job now has all 3,487 molecules, 0 errors.
Checks: AE11_Rn GAS22 non-SCF is identical in jobs 0003 and 0004 (−21872.719410778 Eh); the reference is within 0.003 kcal/mol of
Q-Chem. §2–§9 above remain the 35-set analysis (the prompt's scope). This section is over **36 sets**. The 20 "new" sets now include AE11.

**Mean NER, 36 sets** (it159 SCF is rank **3** of the 16 published, behind ωB97M2 and ωB97M-V):

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 1.001 |
| 2 | ωB97M-V | 1.135 |
| 3 | it159 SCF | 1.139 |
| 4 | it159 non-SCF | 1.141 |
| 5 | ωB97M-V (this work) | 1.144 |
| 6 | it396 non-SCF | 1.147 |
| 7 | **GAS22 SCF** | 1.190 |
| 8 | GAS22 non-SCF | 1.212 |
| 9 | CF22D | 1.231 |
| 10 | ωB97X-V | 1.263 |
| 11 | M06-2X | 1.420 |
| 12 | M052X | 1.547 |
| 13 | MN15 | 1.549 |
| 14 | M08HX | 1.597 |

**20 new sets** (not in MGCDB84; AE11 included). it159 SCF is now rank **2** of 16 (only ωB97M2 is ahead); GAS22 SCF rank 4.

| # | method | mean NER |
|---|---|---|
| 1 | ωB97M2 | 1.080 |
| 2 | it159 SCF | 1.269 |
| 3 | it159 non-SCF | 1.270 |
| 4 | ωB97X-V | 1.276 |
| 5 | ωB97M-V | 1.285 |
| 6 | ωB97M-V (this work) | 1.290 |
| 7 | it396 non-SCF | 1.316 |
| 8 | **GAS22 SCF** | 1.319 |
| 9 | GAS22 non-SCF | 1.344 |
| 10 | CF22D | 1.365 |

**16 MGCDB84-derived sets** are unchanged: ωB97M2 0.903, it396 0.935, ωB97M-V 0.947, ours 0.961, it159 SCF 0.976, GAS22 SCF 1.029.

**AE11 on its own** (absolute energies of 11 heavy atoms; metric in kcal/mol, NER):

| method | metric | NER |
|---|---|---|
| ωB97M-V (ours = published) | 368.9 | 3.154 |
| it159 SCF | 295.7 | 2.528 |
| it396 non-SCF | 283.8 | 2.426 |
| GAS22 SCF | 78.4 | 0.671 |
| ωB97M2 (published) | 410.1 | 3.506 |
| CAMB3LYP (published, best) | 47.4 | 0.405 |

AE11 is the one set where GAS22 is far ahead: its heavy-atom total-energy shift, which cancels in reactions, moves absolute
energies toward the reference. it159 and it396 improve on ωB97M-V there by 0.6–0.7 NER; GAS22 by 2.5.
The density-driven error of AE11_Rn is 0.28 kcal/mol for it159 (0.33 for GAS22).

**Mean NER per data type** (36 sets; only Thermochemistry changes, it now has 17 sets):

| data type | sets | GAS22 SCF | it159 SCF | it396 non-SCF | ωB97M-V (ours) |
|---|---|---|---|---|---|
| Thermochemistry | 17 | 1.142 | 1.151 | 1.126 | 1.177 |

Other types are as in §4.

**Paired comparisons, 36 sets.** ΔNER = A − B per set, negative = A better. Wilcoxon signed-rank; bootstrap of the mean, 10,000
draws, `numpy.random.default_rng(0)`.

| sets | comparison (A vs B) | mean ΔNER (A − B) | median ΔNER | A better | Wilcoxon p | bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| all 36 | GAS22 SCF vs it159 SCF | +0.0516 | +0.0640 | 10/36 | 0.0201 | [-0.0852, +0.1654] |
| all 36 | GAS22 SCF vs ωB97M-V (this work) | +0.0467 | +0.0423 | 10/36 | 0.0249 | [-0.1388, +0.1892] |
| all 36 | GAS22 SCF vs GAS22 non-SCF | -0.0216 | -0.0062 | 27/36 | 0.0004 | [-0.0352, -0.0099] |
| all 36 | it159 SCF vs ωB97M-V (this work) | -0.0049 | -0.0000 | 18/36 | 0.5813 | [-0.0551, +0.0360] |
| all 36 | it159 SCF vs it159 non-SCF | -0.0021 | -0.0014 | 28/36 | 0.0000 | [-0.0030, -0.0013] |
| all 36 | it396 non-SCF vs it159 non-SCF | +0.0060 | +0.0129 | 17/36 | 0.9320 | [-0.0293, +0.0473] |
| all 36 | GAS22 non-SCF vs it159 non-SCF | +0.0711 | +0.0693 | 11/36 | 0.0168 | [-0.0767, +0.1918] |
| 20 new | GAS22 SCF vs it159 SCF | +0.0505 | +0.0493 | 6/20 | 0.0696 | [-0.1976, +0.2434] |
| 20 new | GAS22 SCF vs ωB97M-V (this work) | +0.0297 | +0.0351 | 7/20 | 0.1429 | [-0.2883, +0.2733] |
| 20 new | GAS22 SCF vs GAS22 non-SCF | -0.0247 | -0.0055 | 14/20 | 0.0107 | [-0.0444, -0.0078] |
| 20 new | it159 SCF vs ωB97M-V (this work) | -0.0209 | +0.0081 | 10/20 | 0.7562 | [-0.1060, +0.0449] |
| 20 new | it159 SCF vs it159 non-SCF | -0.0017 | -0.0012 | 14/20 | 0.0083 | [-0.0028, -0.0007] |
| 20 new | it396 non-SCF vs it159 non-SCF | +0.0461 | +0.0347 | 6/20 | 0.0484 | [+0.0002, +0.1075] |
| 20 new | GAS22 non-SCF vs it159 non-SCF | +0.0735 | +0.0693 | 7/20 | 0.0532 | [-0.1815, +0.2733] |

What changes, in plain words:
- **GAS22 SCF vs it159 SCF:** +0.052 over 36 sets (+0.106 over 35). The Wilcoxon test is still significant (p 0.020, it159 better on 26 of
  36 sets, median +0.064), but the bootstrap CI of the mean now includes zero [−0.085, +0.165], because AE11 alone is worth
  −1.86 NER for GAS22. On the 20 new sets: +0.051, p 0.070, CI [−0.198, +0.243]. So "it159 is significantly better than GAS22" holds
  for the rank-based test and for 35 sets, but is **not robust** to including AE11.
- **it159 SCF vs ωB97M-V (ours):** −0.005 (p 0.58): the two are tied; it159 is now fractionally ahead (1.139 vs 1.144) instead of fractionally behind.
- **GAS22 SCF vs ωB97M-V:** +0.047 (p 0.025, CI [−0.139, +0.189]).
- **SCF vs non-SCF:** unchanged: GAS22 −0.022 (p 0.0004), it159 −0.002 (p < 0.0001, 28 of 36 sets better, tiny).
- **it396 vs it159 non-SCF:** +0.006 (p 0.93); on the 20 new sets +0.046 (p 0.048, CI [+0.000, +0.108]), it396 still worse there.
- Order on 36 sets (SCF for it159/GAS22): it159 1.139 < ωB97M-V 1.144 < it396 1.147 < GAS22 1.190. On the 35 sets it was
  1.086 (ωB97M-V) < 1.099 < 1.110 < 1.205.

**SCF quality over 3,487 molecules.** GAS22 density-driven error mean 0.150, median 0.121, max 1.915 kcal/mol (10 above 1);
it159 mean 0.039, median 0.031, max 1.293 (1 above 1). Unconverged candidate SCF: GAS22 66, it159 15 (unchanged; the AE11_Rn
SCFs converged by DIIS). it159's AE11 density-driven errors are the 3rd–5th largest of its run (Hg 0.34, Yb 0.33, Rn 0.28).

**Caveat on the new numbers.** AE11 enters as one of 36 equally weighted sets, and it is a set of absolute total energies. A
functional's AE11 score says little about reaction energies, which most other sets measure. Because AE11 is one set with a
very large NER for ωB97M-V-like functionals (3.2), the 36-set means and the bootstrap intervals are more sensitive to it
than to any other set. Both the 35-set and the 36-set results should be reported.

## 12. Convergence fix (2026-10-24): unconverged records reworked

The user asked for the unconverged SCFs of the three functionals with an SCF (ωB97M-V reference, it159, GAS22) to be converged for credibility.
Details, evidence and scripts: `runs/convergence_fix/README.md`. No project code was changed.

| | before | after |
|---|---|---|
| ωB97M-V reference SCF unconverged | 2 | **0** |
| it159 candidate SCF unconverged | 15 | **0** |
| GAS22 candidate SCF unconverged | 66 | **2** (DAPD_Pd and 3d4dIPSS_Pd_GS: the same Pd atom) |

**Cause.** The torch potentials of GAS22 and it159 are numerically non-smooth in the far minority-spin tail (ρσ ≈ 1e-10, reduced gradient ≈ 1e9): a tiny orbital
rotation changes that tail strongly and the potential jumps. DIIS then stalls with |g| ≈ 1e-5 against the 1e-6 criterion, while the energy is stationary
to ~1e-9 Eh. The libxc ωB97M-V potential is smooth, and torch-ωB97M-V converges (8 cycles on Na), so the GPU/torch machinery is sound; the cause is in the GAS22/it159 forms.

**What was done, by class** (stored per record in `cand.convergence`):

| class | meaning | GAS22 | it159 |
|---|---|---|---|
| strict | DIIS |g| < 1e-6, ΔE < 1e-10 (spin threshold ρσ 1e-14 instead of 1e-10) | 10 | 3 |
| strict_gradient | an iterate has |g| < 1e-6 (energy criterion missed by noise) | 2 | 0 |
| noise_limited | best iterate |g| < 1e-3 and the 5 best iterates agree within 1e-8 Eh (1e-7 for 3 Na–Li records, 1e-6 for K) | 52 | 12 |
| unconverged | Pd atom, |g| ≈ 2.5e-2 for every solver setting; energy reproducible to ~1e-6 Eh | 2 | 0 |

Noise-limited records: largest |g| 4.2e-4 (GAS22), 7.9e-6 (it159); largest energy spread 8.3e-8 Eh = 5e-5 kcal/mol. **These records are not converged to the
strict 1e-6 gradient criterion; they are converged to the energy.** The class is in the record, so both views can be reported: of the 66 GAS22 records, 12 are strictly converged (10 strict + 2 strict_gradient), 52 are noise-limited and 2 unconverged;
of the 15 it159 records, 3 are strictly converged and 12 noise-limited. Every record except the two Pd ones meets a stated energy tolerance: the 5 best iterates agree within 8.3e-8 Eh (5e-5 kcal/mol).

**Reference SCFs.** Pol130_CH2PH2- (with field) and RG10N_KrKr_3p900 were recomputed on the CPU: both converge by DIIS (11 and 8 cycles), ωB97M-V − Q-Chem 6e-5 and
3e-4 kcal/mol (before 2.5e-3 and 3e-4). The stored orbitals were replaced (old files backed up) and the candidate part of those two molecules was recomputed in all four jobs.

**Effect on the results.**
- Largest change of a corrected SCF energy: 3.9e-4 kcal/mol (GAS22, median 6e-5), 1.8e-4 (it159, median 4e-5). E_ref of the two references changes by −0.0025 and +0.00004 kcal/mol.
- Mean NER changes by ≤ 1e-5 for every method (36 sets: it159 SCF 1.13872, GAS22 SCF 1.19031, ωB97M-V ours 1.14365). No ranking, rank or paired test changes. So the earlier conclusion
  that the unconverged records did not drive the results is now measured, not assumed.
- Density-driven error unchanged: GAS22 mean 0.150 / max 1.915, it159 mean 0.0385 / max 1.293 kcal/mol.

**What remains.** The Pd atom under GAS22 (2 of 3,487 records, 0.06 %): the SCF never settles (|g| ≈ 2.5e-2, energy within ~1e-6 Eh across five solver settings and
across the GPU and CPU paths; |E_SCF − E_nonSCF| = 0.049 kcal/mol). It enters 3d4dIPSS and DAPD. A change of 1e-5 kcal/mol is far below those sets' errors (several kcal/mol), but the record stays flagged.
