# GSCDB evaluation of our functionals: project details

Folder: `/home/ash/Documents/dft_fucntional_sr_gold_std`. Last updated 2026-10-24.

This file is the summary. The detailed operating notes and the fix log are in `GSCDB_NOTES.md`; per-job reports are
in `runs/jobNNNN_*/REPORT.md`; the project-wide record is in
`/home/ash/Documents/dft_functional_sr/project_master_details.md` (§13g–§13j).

## 1. Goal and scope

The goal is to test the functionals found by the MGCDB84 search on GSCDB137 (Liang & Head-Gordon, JCTC 21, 12601
(2025); `ct5c01380.pdf`), self-consistently, and to rank them against the 16 functionals whose GSCDB energies are
published.

Scope so far is a cost-limited subset: the 36 sets whose largest molecule has ≤ 500 basis functions. That is
2,734 entries and 3,487 molecules. **All four jobs now have all 3,487 molecules**: AE11_Rn cannot be computed on the GPU, so it
was added afterwards on the CPU (§4). Results are given on **35 sets** (without AE11, the original scope, §2) and on **36 sets**
(with AE11, §2b). AE11 changes the GAS22 comparison a lot (§2b). The claim is scoped to
this subset, not to GSCDB137. With the 16 published functionals the subset ranking agrees with the full GSCDB ranking
only partly (Spearman 0.93, Kendall 0.78).

Rules for every run:
- reporting only (nothing is fitted or selected on GSCDB);
- explicit user approval per job;
- systemd user units only, never nohup or setsid;
- the stored reference orbitals are changed only through the runner (one exception, approved: `AE11_Rn.h5` was replaced by the
  CPU-computed orbitals on 2026-10-24).

## 2. Results (mean NER, 35 sets without AE11; lower is better)

| method | mode | all 35 | 19 new sets | 16 MGCDB84-derived | rank among the 16 published (all 35) | job |
|---|---|---|---|---|---|---|
| ωB97M2 (published) | | 0.930 | 0.953 | 0.903 | — | |
| ωB97M-V (published) | | 1.077 | 1.186 | 0.947 | — | |
| ωB97M-V (this work) | SCF | 1.086 | 1.191 | 0.961 | — | 0001 |
| **it159** | SCF | **1.099** | 1.202 | 0.976 | 3 | 0001 |
| it159 | non-SCF | 1.101 | 1.204 | 0.979 | 3 | 0001 |
| it396 | non-SCF | 1.110 | 1.258 | 0.935 | 3 | 0002 |
| ωB97X-V (published) | | 1.197 | 1.155 | 1.247 | — | |
| **GAS22** | SCF | **1.205** | 1.353 | 1.029 | 4 | 0004 |
| GAS22 | non-SCF | 1.227 | 1.379 | 1.047 | 4 | 0003 / 0004 |
| CF22D (published) | | 1.233 | 1.375 | 1.063 | — | |

Paired tests over sets:
- **it159 vs ωB97M-V:** ΔNER +0.013, p 0.39. Indistinguishable.
- **it396 vs it159:** equal overall (p 0.78), but significantly worse on the new sets (+0.054, p 0.018). Its
  MGCDB84 validation gains do not transfer.
- **GAS22 vs it159:** significantly worse overall, non-SCF (+0.126, p 0.005) and SCF (+0.106, p 0.006; new sets
  +0.151, p 0.016). GAS22 is worst on transition metals (2.17) and thermochemistry (1.17). This reverses the MGCDB84
  test order, where GAS22 was best with 3.600. SCF improves GAS22 by 0.022 NER (p 0.0003); it159's SCF effect was 0.002.
- **Order on the 35 sets (SCF):** ωB97M-V ours 1.086 < it159 1.099 < it396 1.110 (non-SCF) < GAS22 1.205.

## 2b. Results with AE11 (mean NER, all 36 sets; AE11_Rn computed on the CPU for every job, 2026-10-24)

| method | mode | all 36 | 20 new sets | 16 MGCDB84-derived | rank among the 16 published (all 36) | job |
|---|---|---|---|---|---|---|
| ωB97M2 (published) | | 1.001 | 1.080 | 0.903 | — | |
| ωB97M-V (published) | | 1.135 | 1.285 | 0.947 | — | |
| **it159** | SCF | **1.139** | 1.269 | 0.976 | 3 | 0001 |
| it159 | non-SCF | 1.141 | 1.270 | 0.979 | 3 | 0001 |
| ωB97M-V (this work) | SCF | 1.144 | 1.290 | 0.961 | — | 0001 |
| it396 | non-SCF | 1.147 | 1.316 | 0.935 | 3 | 0002 |
| **GAS22** | SCF | **1.190** | 1.319 | 1.029 | 3 | 0004 |
| GAS22 | non-SCF | 1.212 | 1.344 | 1.047 | 3 | 0003 / 0004 |
| CF22D (published) | | 1.231 | 1.365 | 1.063 | — | |

AE11 on its own (NER): ωB97M-V 3.15, it159 2.53, it396 2.43, **GAS22 0.67**, ωB97M2 3.51, best published CAMB3LYP 0.41.
GAS22's heavy-atom total-energy shift (cancels in reactions) brings absolute atomic energies much closer to the reference.
Paired tests over 36 sets (ΔNER = A − B, negative = A better; details in `runs/job0004_GAS22_SCF_datasets/REPORT.md` §11):
- **GAS22 SCF vs it159 SCF:** +0.052, Wilcoxon p 0.020 (it159 better on 26/36 sets), but the bootstrap CI [−0.085, +0.165]
  includes zero. "Significantly worse" holds on 35 sets (+0.106, p 0.006) and rank-based on 36, not for the mean.
- **it159 SCF vs ωB97M-V (ours):** −0.005, p 0.58: tied, it159 fractionally ahead (it was fractionally behind on 35 sets).
- **GAS22 SCF vs ωB97M-V:** +0.047, p 0.025, CI [−0.139, +0.189].
- **it396 vs it159 non-SCF:** +0.006 (p 0.93); on the 20 new sets +0.046 (p 0.048).
- On the 20 new sets it159 SCF is rank 2 of the 16 published (1.269; only ωB97M2 1.080 is ahead).
- Order on 36 sets: it159 1.139 < ωB97M-V 1.144 < it396 1.147 < GAS22 1.190.

## 3. Physical constraints (checked 2026-10-23)

These are the project's constraint tests (`dftsr/constraints.py`, after Duston et al. 2026, Sec. 1.3). They were
run with the published or stored parameters of each functional.

| constraint (threshold) | ωB97M-V | it159 | it396 | GAS22 |
|---|---|---|---|---|
| spin symmetry, O₂ triplet grid (< 1e-5 Ha) | pass (1e-15) | pass (1e-15) | pass | pass |
| UEG exchange limit g_x = 0.85 at r_s 0.5–5 (±0.5 %) | pass (0.85) | pass (0.85) | pass (0.85) | **fail (0.862)** |
| exchange uniform-coordinate scaling, H₂O, λ = 0.5/2/5 (< 1e-4 Ha) | pass (4e-15) | pass (4e-15) | pass | pass |
| grid sensitivity, AE18 atoms, (99,590) vs (250,974) (< 0.015 kcal/mol) | pass (4.4e-5) | pass (8.9e-5) | pass (1.3e-4) | pass (1.6e-4) |
| violations | 0 | 0 | 0 | 1 |

- **ωB97M-V satisfies every constraint that it159 satisfies.** it159 is built in the ωB97M-V frame: same exchange
  form, VV10, ω = 0.3 and UEG normalisation. So it inherits these properties by construction.
- GAS22's UEG violation (g_x = 0.862) fits the large systematic total-energy shifts it shows for heavy all-electron
  atoms against it159 (Hg −888, Kr −236 kcal/mol). These shifts cancel in reaction energies.
- The suite checks only these four. Other exact conditions were not tested for any functional: the Lieb–Oxford
  bound, the correlation UEG limit and scaling, one-electron self-interaction freedom, the gradient expansion.

## 4. Technical state and lessons (details in `GSCDB_NOTES.md`)

**Protocol.**
- ωB97M-V reference SCF in the GSCDB setup, with a state search against Q-Chem's ωB97M-V energy. Its orbitals are
  stored in `/media/ash/PortableSSD/dft_sr/gscdb_ref_wb97mv/` and are complete and validated for all 3,486 molecules.
- Candidates are evaluated non-SCF on that density and/or SCF starting from those orbitals.
- The runner is `scripts/gscdb_scf.py`. Non-SCF-only mode is `--nonscf-only` or a `NONSCF_ONLY` file in the job
  folder (~7 GPU-h per functional).

**GPU4PySCF 1.8.1 defects for h shells (l = 5).** These should be reported upstream.
1. Point-charge nuclear attraction is wrong by up to 10 Eh. Workaround: CPU core Hamiltonian for every l ≥ 5
   molecule.
2. The fused range-separated `get_k` (lr/sr factors) is wrong. Workaround: split into full-range and long-range K.
3. The semilocal XC matrix is wrong for all-electron Rn (largest exponent 2.3e8). No GPU workaround, so AE11_Rn is
   pre-excluded in the GPU runs. **Workaround for job 0004 (2026-10-24):** `runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py`
   runs the whole protocol on the CPU (validated on AE11_Kr to 5e-6 kcal/mol; AE11_Rn ωB97M-V − Q-Chem +0.003 kcal/mol, 61 s).
   The stored `AE11_Rn.h5` is now the correct CPU one (old divergent file backed up in the job 0004 folder). The same script
   (`--ref-from-store`, `--nonscf-only`) then added AE11_Rn to jobs 0001, 0002 and 0003 (11–34 s each), so every job has all
   3,487 molecules. A GPU run on AE11_Rn would still be wrong.

**Other fixes:**
- the VV10 SG-1 guard uses the true Z;
- CPU stability analysis now gets a NumPy core Hamiltonian;
- the stability re-SCF has fallbacks;
- the state search can use a field-sibling guess.

The MGCDB84 code paths were not touched. MGCDB84 uses def2-QZVPPD, where lmax = 4 for every element, so it is not
affected by these defects.

**Agreement with Q-Chem.**
- Our ωB97M-V agrees with Q-Chem's published ωB97M-V with a median of 3.6e-4 kcal/mol.
- On the 29 sets without state differences, our mean NER is 1.077 against 1.078 published.
- State differences: W4-17_o3 +2.9, ORBH35_CO3B_S04T +8.4, IP23_C2_n −3.4, 3d4dIPSS_Sc_ES −13.6, and a few 3d/4d
  atoms at 0.1–0.4 kcal/mol. They affect all our functionals alike.

**SCF behaviour of the torch candidates.**
- The Newton fallback is not implemented for TorchNumInt (`NotImplementedError`), so unconverged candidates end
  after 150 DIIS + 300 level-shift cycles.
- That is slow (30–45 min for Ar/Kr dimers and some Pol130/Dip146 field points; 3.5 h for two Kr–Kr dimers), but the
  energies were fine: |E_SCF − E_nonSCF| ≤ 0.12 kcal/mol for it159 (15 unconverged) and ≤ 0.84 for GAS22 (66 unconverged,
  1.9 %: light open-shell atoms, 3d4d atoms, field points). GAS22's density-driven error is mean 0.150 / max 1.92
  kcal/mol, about four times it159's (0.038 / 1.29).
- **Convergence fix (2026-10-24, `runs/convergence_fix/README.md`).** Cause: the GAS22 and it159 torch potentials are non-smooth in the
  far minority-spin tail (ρσ ≈ 1e-10), so the SCF gradient stalls at ~1e-5 while the energy is stationary to ~1e-9 Eh (libxc
  ωB97M-V and torch-ωB97M-V do not have this). Result: ωB97M-V references 2 → 0 (recomputed on the CPU, orbitals replaced), it159
  15 → 0 (3 strict, 12 noise-limited), GAS22 66 → 2 (the Pd atom, twice; |g| stuck at 2.5e-2, energy reproducible to ~1e-6 Eh).
  GAS22: 12 strictly converged, 52 noise-limited (best iterate |g| < 1e-3 and energy of the 5 best iterates within ≤ 8.3e-8 Eh).
  Energies moved by ≤ 3.9e-4 kcal/mol; mean NER by ≤ 1e-5; no ranking or test changed. Class stored in `cand.convergence`.

**Cost per functional on this subset** (reference orbitals stored):
- non-SCF: ~7 GPU-h, ~2.5 h wall;
- SCF: it159's candidate part 50.9 GPU-h; GAS22 (job 0004) 65.3 GPU-h, 21.8 h wall on 3 GPUs, of which 22.5 GPU-h were
  the slowest 1 % of molecules (34).
- Job 0001 cost 138 GPU-h only because it also computed and recomputed the references.

## 5. Jobs

| job | functional | mode | dates | GPU-h | report |
|---|---|---|---|---|---|
| 0001 | it159 | SCF (+ references); AE11_Rn added on the CPU 2026-10-24 | 2026-10-20 → 10-23 | 138.2 | `runs/job0001_it159_datasets/REPORT.md` |
| 0002 | it396 | non-SCF; AE11_Rn added 2026-10-24 | 2026-10-23 | 7.4 | `runs/job0002_it396_datasets/REPORT.md` |
| 0003 | GAS22 (published) | non-SCF; AE11_Rn added 2026-10-24 | 2026-10-23 | 7.3 | `runs/job0003_GAS22_datasets/REPORT.md` |
| 0004 | GAS22 (published) | SCF, 3,487 molecules (AE11_Rn on the CPU) | 2026-10-23 → 10-24 | 65.3 + 0.02 | `runs/job0004_GAS22_SCF_datasets/REPORT.md` |

Prompts: `IT159_GSCDB_LE500_PROMPT.md` (root), the prompts inside the job folders, and
`GAS22_GSCDB_LE500_SCF_PROMPT.md` (root; run as job 0004).

## 6. Open items (each needs the user's go-ahead)

- **Torch-functional SCF convergence:** done for the records (see §4, convergence fix): 2 GAS22 Pd records remain unconverged. A
  smooth treatment of the minority-spin tail in the functional code (not done, code change) would make the SCF converge strictly and
  remove most of the slow tail.
- **Report both scopes** (35 sets without AE11, 36 sets with it): AE11 moves GAS22 from +0.106 to +0.052 behind it159.
  Done: AE11_Rn for all jobs (§2b).
- **Extension:** to the sets whose largest molecule has ≤ 1000 basis functions (pair-winner flip rate 1/10), or
  ≤ 2000.
- **MCP:** a restart of the MCP server, so that `gscdb_report` uses the updated `finalize` for non-SCF-only jobs.
  A new Claude session does this automatically.
- **Upstream:** a report to GPU4PySCF about the three h-shell defects.

## 7. Is it159 superior and publishable? (assessment 2026-10-24, after job 0004)

This is an assessment of what the data support, not a decision. Whether and how to publish is for the user and
co-authors. All numbers are mean NER over the 35 complete sets unless stated, from `runs/job000{1,2,4}_*/REPORT.md`.

**Short answer.** it159 is clearly better than GAS22 on this subset and ties with ωB97M-V. "Superior" without
qualification is not supported. "Transfers to GSCDB at the level of ωB97M-V" is supported.

**Update after AE11 was added for every job (2026-10-24, §2b).** The statements below are written for the 35 sets. With
the 36 sets: it159 SCF 1.139 vs ωB97M-V 1.144 (tied, p 0.58; it159 now fractionally ahead), it159 is rank 2 of 16 on the 20
new sets, and the gap to GAS22 shrinks to +0.052 (Wilcoxon p 0.020, bootstrap CI [−0.085, +0.165] includes zero). So
"it159 is significantly better than GAS22" is **not robust** to including AE11, where GAS22 is far ahead (NER 0.67 vs
2.53 for it159). "Tied with ωB97M-V" and "transfers" are unchanged. A paper should report both scopes and say why AE11
(absolute atomic energies, the one set GAS22 wins) matters; the CI for the GAS22 comparison should not be quoted as excluding zero.

### What the results support
- **vs GAS22 (35 sets; see the 36-set update above):** significantly better, SCF against SCF. Over all 35 sets GAS22 SCF is worse by +0.106 NER (p 0.006,
  bootstrap CI [+0.035, +0.190]); over the 19 new sets by +0.151 (p 0.016). Non-SCF gives the same verdict (+0.126,
  p 0.005). GAS22 SCF wins no set among the 22 methods in the per-set table.
- **vs it396:** equal overall (+0.009, p 0.78). it159 is significantly better on the 19 new sets (it396 is worse by
  +0.054, p 0.018). it396's MGCDB84 validation gains did not transfer.
- **vs the 16 published functionals:** rank 3 on all 35 sets (it159 SCF 1.099), rank 4 on the 19 new sets (1.202).
  Only ωB97M2 (0.930, a double hybrid) and ωB97M-V (1.077, the functional it is built from) are ahead overall.
  On the 16 MGCDB84-derived sets it is rank 3 (0.976).
- **Transfer:** a functional found on MGCDB84 still performs near the best of its class on a different benchmark.
  On MGCDB84 it improves WRMSD_tot over ωB97M-V by 7.9 % (3.767 vs 4.090, SCF; `project_master_details.md` §13k).
- **Physical constraints** (§3): it159 passes all four checks, as ωB97M-V does. GAS22 fails the UEG-exchange limit
  (g_x 0.862).
- **Protocol:** GSCDB was used for reporting only. Nothing was fitted or selected on it. Our ωB97M-V agrees with Q-Chem
  (median 3.6e-4 kcal/mol; 1.077 vs 1.078 published on the 29 unaffected sets), so the pipeline is validated.

### What they do not support
- **"Better than ωB97M-V":** it159 is not. ΔNER = +0.013 (p 0.39, CI [−0.018, +0.043]); on the 19 new sets +0.011
  (p 0.47). The point estimate slightly favours ωB97M-V. it159 is better on noncovalent sets (0.871 vs 0.931), similar on
  thermochemistry and barrier heights, and worse on transition metals (1.569 vs 1.459), electric-field properties
  (1.454 vs 1.398), V30 (1.116 vs 0.926) and isomerization (1.322 vs 1.166).
- **A broad GSCDB claim:** the subset is 35 of 137 sets (size-restricted, no large noncovalent complexes, few barrier
  heights). The 35-set results leave out AE11, which strongly favours GAS22 (NER 0.67 vs 2.53 for it159, 3.15 for ωB97M-V),
  so the GAS22 gap is partly an artefact of leaving it out (§2b gives the 36-set numbers). Its ranking agrees with the full-GSCDB ranking only partly (Spearman 0.93, Kendall
  0.78); for close pairs the winner flipped in 4 of 10 cases, and it159 vs ωB97M-V is such a pair.
- **A strictly converged SCF comparison:** after the convergence fix (§4) all it159 records and all but 2 GAS22 records (Pd atom, 0.06 %)
  are converged; 3 of 15 it159 and 12 of 66 GAS22 repaired records meet the strict 1e-6 gradient criterion, the rest are converged
  to the energy (5 best iterates within ≤ 8.3e-8 Eh) with an orbital gradient floor of ≤ 4.2e-4. The repair moved energies by ≤ 4e-4
  kcal/mol and mean NER by ≤ 1e-5, so the earlier worry is resolved: the unconverged records did not drive any result.
- **Full independence:** it159 came from a search on MGCDB84. The GAS22 and it396 comparisons rest on three runs that
  share the same 36 sets, and GAS22 still has the best MGCDB84 test WRMSD (3.600 vs 3.818); only on this subset is the
  order reversed (WCPT6, the one test-role set, keeps it).

### What a paper could and could not claim
- **Supported:** "it159 transfers to GSCDB at the level of ωB97M-V, improves MGCDB84 error by about 8 %, and is more
  robust than GAS22 and it396." Suitable as a validation result.
- **Not supported:** "it159 is a better general-purpose functional than ωB97M-V." On GSCDB the two are tied.

### What reviewers would likely ask for (all need the user's go-ahead)
1. The full GSCDB, or at least the ≤ 1000-basis-function extension (new reference orbitals; estimate the cost first).
2. AE11 back in. Done for all jobs with the CPU script (§2b); the remaining reviewer point is a GPU-independent check of that path.
3. SCF convergence: done for the records (§4); a reviewer may still ask for strict 1e-6 convergence everywhere, which needs a smoother
   minority-spin tail in the functional code (the Pd atom under GAS22 stays unconverged).
4. Upstream report of the three GPU4PySCF h-shell defects (point-charge nuclear attraction, fused range-separated
   `get_k`, semilocal XC for all-electron Rn), since GSCDB molecules with h functions depend on our workarounds.
5. (done) The unconverged-record sensitivity: measured by the repair itself, mean NER ≤ 1e-5.
