# Project master details: agentic XC-functional discovery on MGCDB84

*Snapshot written 2026-10-13 ~12:40 local time, during the initial setup session. This document records the structure, every design decision, every measured timing/accuracy number, the bugs found and the open items. `README.md` is the short user guide; this file is the complete record.*

---

## 1. Goal (as specified by the user)

1. Build an **MCP server** that tests new DFT exchange–correlation functionals **non-self-consistently (non-SCF)**, like Duston et al. 2026 (`agentic_discovery_paper_2026.pdf`), so that Claude Code ("the full force of agentic AI") can search for a new, promising functional for the dataset used in Ma et al. 2022 (`evolving_symbolic_2022.pdf`). Both papers use **MGCDB84**.
2. Write a **document of the older functional forms** from the two papers → `docs/FUNCTIONAL_FORMS.md`.
3. The MCP server must **record every iteration** (each new functional form tried), with its **accuracy** and the **number of tokens consumed** at that iteration.
4. **Attach the MCP server to the local Claude Code** → `.mcp.json` (project scope).
5. **Estimate the time** to evaluate one new functional non-SCF on this machine.
6. **Do not change CUDA or NVIDIA** on the system.
7. **Download the packages for SCF evaluation** (for later). SCF evaluation is a separate MCP tool, and only a functional that is successful non-SCF goes on to full SCF on the dataset. **Estimate that time** too.
8. **Write the prompt for the non-SCF search** → `prompts/NONSCF_SEARCH_PROMPT.md`.
9. **Compute the reference densities** of the original dataset with SCF and save them to a new folder on `/media/ash/PortableSSD`. **Estimate that time** too. Status: running.

The user's original message is preserved in `prompts.txt`, which the user created; it is not generated.

---

## 2. Machine and environment

| item | value |
|---|---|
| OS | Linux 6.8.0-138-generic, user `ash` |
| CPU / RAM | 64 cores, 503 GB RAM (≈ 405 GB of it page cache at the start) |
| GPUs | 3× **NVIDIA RTX 6000 Ada**, 49 GB each. Driver **580.178.04**, CUDA runtime 13.0. System toolkit `/usr/local/cuda-12.8` (untouched). |
| GPU roles | GPU 0: no display. **GPU 1: Xorg + gnome-shell (main display)**. GPU 2: Xorg. Heavy compute on GPUs 1 and 2 may make the desktop lag. |
| SSD | `/dev/sda1` mounted at `/media/ash/PortableSSD`, **exFAT**, 1.9 TB, **1 MB cluster size**: every small file occupies ≥ 1 MB. For example, the 10,049 tiny ACCDB xyz files show as 9.9 GB. Prefer few, large files. |
| System Python | 3.10.12 (numpy 1.21.5, scipy 1.8.0, torch 2.7.1+cu126). Not used by the project. |
| Project venv | `/home/ash/Documents/dft_functional_sr/.venv`, built with **uv 0.12.21**, **Python 3.12.14** |
| Venv packages | **pyscf 2.14.0**, **gpu4pyscf 1.8.1** (`gpu4pyscf-cuda13x`), **cupy 14.2.0**, **torch 2.14.1+cu130**, pyscf-dispersion, geometric, h5py, pandas, numpy, scipy, **mcp 1.30.0** (pinned `mcp[cli]<2`) |
| CUDA policy | All CUDA runtime libraries come from **pip wheels inside `.venv`** (nvidia-* wheels pulled in by torch/cupy). **Nothing system-level was installed or modified.** |
| Claude Code | `/home/ash/.local/bin/claude`, version 2.1.296 |

Install command used:
```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy h5py pandas "mcp[cli]" pyscf pyscf-dispersion geometric torch
uv pip install --python .venv/bin/python gpu4pyscf-cuda13x
uv pip install --python .venv/bin/python "mcp[cli]<2"     # 2.x renamed FastMCP -> MCPServer
```

---

## 3. Directory structure

### 3.1 Project (`/home/ash/Documents/dft_functional_sr/`)
```
agentic_discovery_paper_2026.pdf          Duston et al., arXiv:2605.05460 (SAFS26)
evolving_symbolic_2022.pdf                Ma et al., Sci. Adv. 8, eabq0279 (SyFES / GAS22)
gold_standard_dataset_2508.13468v1.pdf    Liang & Head-Gordon, GSCDB137 (future extension)
Symbolic Regression for Discovering ... .pdf   (not used yet)
prompts.txt                               user's original request (user file)
README.md                                 short user guide
project_master_details.md                 this file
.mcp.json                                 registers MCP server "dft-functional-search" (project scope)
.venv/                                    Python 3.12 environment

dftsr/                    core library (~2850 lines incl. scripts/server)
  __init__.py
  config.py               paths/settings (env-overridable)
  xc_torch.py             PyTorch B97-family building blocks + reference ωB97M-V/B97M-V
  api.py                  candidate-functional API (Functional base class, helpers, loader)
  dataset.py              MGCDB84 parsing, SyFES splits, WRMSD weights, geometry resolution
  features.py             density HDF5 -> per-split float32 feature caches; SplitData (GPU)
  evaluator.py            non-SCF energies, ReactionSet, L-BFGS fit, float64 metrics
  constraints.py          4 physical/technical constraint tests
  worker.py               evaluation pipeline, smoke test, persistent GPU worker, CLI
  tracker.py              SQLite iteration DB, token accounting from Claude Code transcripts
  scf.py                  TorchNumInt: candidate XC injected into GPU4PySCF (autograd vxc)

mcp_server/server.py      FastMCP server (15 tools), GPU worker pool
baselines/
  wb97mv.py               ωB97M-V semilocal part (seed/baseline), family 'rsh'
  gas22.py                GAS22 (published coefficients as init), family 'rsh'
  b97mv.py                B97M-V, family 'mgga'
docs/FUNCTIONAL_FORMS.md  all prior functional forms + anti-patterns + paper numbers
prompts/NONSCF_SEARCH_PROMPT.md   agent prompt for the non-SCF search (also served by get_task_brief)
scripts/
  compute_densities.py        ωB97M-V density campaign worker (resumable, lock files)
  run_density_campaign.sh     launch one detached worker per GPU
  density_progress.py         progress / errors / unconverged report
  compute_constraint_data.py  O2 / H2O / AE18 two-grid data for the constraint tests (CPU PySCF)
  build_feature_cache.py      build GPU caches (--allow-partial for testing)
  establish_baseline.py       ωB97M-V baseline, J_target, seed iterations
  post_density_pipeline.sh    waits for densities -> caches -> baseline -> test cache
  scf_evaluate.py             full SCF evaluation of a candidate (multi-GPU, resumable)
candidates/               every evaluated candidate file, <iteration id>_<name>.py (it4–it400); README.md = why each was tried
runs/                     iterations.sqlite (created on first MCP start; currently empty dir)
```

### 3.2 Data (`/media/ash/PortableSSD/dft_sr/`, override with `DFTSR_DATA_ROOT`)
```
external/
  ACCDB/                    git clone --depth 1 github.com/peverati/ACCDB (11 GB on exFAT)
    Databases/MGCDB84/DatasetEval.csv   reactions (Hartree), MGCDB84_ALL.list (5931 molecules), *.list per subset
    Geometries/*.xyz        10,049 xyz files (line 2 = "charge multiplicity")
  GSCDB138/                 git clone --depth 1 github.com/JiashuLiang/GSCDB138 (46 GB; Q-Chem inputs, xyz, Info/DatasetEval.csv)
  syfes_src/gr/symbolic_functionals/   sparse checkout of google-research (SyFES code: splits, weights, xc, GAS22 colab)
densities/
  wb97mv_def2qzvppd_L3/     <molecule>.h5 per molecule + _locks/ + _logs/ (*.jsonl per GPU, worker_gpu*.out)
  constraints/constraints.h5
feature_cache/wb97mv_def2qzvppd_L3/{train,validation,test}/   (currently a PARTIAL test build; rebuilt automatically)
scf_runs/                   SCF evaluation jobs (job<id>_it<iteration>/)
post_density.log            log of post_density_pipeline.sh
```

---

## 4. Papers: key facts used

### 4.1 SyFES / GAS22 (Ma, Narayanaswamy, Riley, Li 2022)
* In the PDF text extraction **"ω" is dropped**: "B97M-V" in the text means **ωB97M-V**. The GAS22 colab uses `dft.libxc.rsh_coeff('wb97mv')` and `f_rsh`, and the stated ωB97M-V term selection matches. Both papers therefore start from **ωB97M-V** with ω = 0.3, 0.15 SR-HF, VV10.
* Regularized evolution: population 100, tournament 25; 12 evolutions; 29,628 forms explored; 50 GPU workers. CMA-ES fitting with 5 restarts and parameters in [−10, 10]. Functional-equivalence checking speeds up the search ~10×.
* Instruction set: s = p+q, p−q, p×q, p/q, s += p×q, pⁿ (n ∈ {2, 3, 4, 6, 1/2, 1/3}), p/(1+γp).
* Rediscovered B97 exchange in < 4000 mutations (TCE subset, 568/189/190 split).
* Densities: **ωB97M-V, def2-QZVPPD, PySCF default grid, SG-1 prune for VV10.** Of the 5931 single points, 7–8 did not converge, affecting 6–7 data points (< 0.01 % change).
* Split: train 870 / validation 2960 / test 1150 (= ωB97M-V "training / primary test / secondary test").
* Weights: TCD 0.1; TCE, AE18 1; NCD, ID, BH 10; NCED, NCEC, IE 100; RG10 10000.
* GAS22 test WRMSD 3.585 vs 4.237 for ωB97M-V (−15 %). SCF ≈ non-SCF. Worse than ωB97M-V only on TCD.
* Code DOI: zenodo 6767222. MGCDB84 SI: doi 10.1080/00268976.2017.1333644.

### 4.2 SAFS26 (Duston et al. 2026, ByteDance Seed)
* LoongFlow-based Plan → Execute → Summarize loop. 4 islands, elite archive, 80/20 exploration/exploitation, periodic migration, per-lineage retrieval tools, global dead-end synthesis. 400+ iterations per run. JAX executor; L-BFGS fit on train; scored on validation.
* Score: R = J_target / WRMSD_val × 0.9^n_violations, with **J_target = 3.45 kcal/mol** (≈ 15 % below the ωB97M-V validation value of 4.02).
* Constraints: spin symmetry (O2 triplet, < 1e-5 Ha); UEG exchange g_x(s=0, t=k_c) = 0.85 within 0.5 %, at 4 densities; uniform coordinate scaling of exchange (H2O, λ = 0.5, 2, 5, < 1e-4 Ha); AE18 grid (99,590) vs (250,974) < 0.015 kcal/mol.
* Final metric "tot": validation + test excluding RG10, 3547 points. Two C20C24 molecules were moved test → validation.
* Results: see table §10.3 and `docs/FUNCTIONAL_FORMS.md`.
* Lessons: unconstrained search cheats (antisymmetric ζ term, sigmoid layer with bias in exchange, Laplacian feature, steep r_s switch). Frozen-parameter grafting overfits validation (val 3.64, test 4.09). Validation leakage grows with iterations (800 vs 400).

### 4.3 Can SAFS26 be evaluated non-SCF here? (checked 2026-10-19)
* **The paper gives no fitted parameters** for SAFS26-a or SAFS26-b. It does not mention a code or data release. So, unlike published GAS22, *their* functionals cannot be run in our pipeline. Exact values would need the authors (ByteDance Seed / Princeton).
* **SAFS26-a (§2.4, Eqs. 7–14): the form is reconstructable and refittable.** The parameter count matches the paper's 50:
  * exchange 4: the ωB97M-V P_x (c₁ w, c₂ u; c_x,0 = 0.85 fixed) + c_v,0 v + c_v,1 v², with v = st/(1+st+ε);
  * same-spin 11 + 11: numerator = 5 ωB97M-V P_css terms + c_v v, v² + tanh(c_z0 z + c_z1 z²) + c_x χ, χ²; denominator 1 + Σ softplus(d_i) φ_i over {5 P_css monomials, v, v², z, z², χ, χ²};
  * opposite-spin 12 + 12: the same pattern on the 6-term ωB97M-V P_cos;
  * fixed: γ_x, γ_css, γ_cos and the ωB97M-V power templates;
  * descriptors: z = f_ζ s t/(1 + f_ζ s t + ε), χ = z s t/(1 + z s t + ε); per-spin s, t for same-spin, spin-averaged for opposite-spin. SAFS uses s = |∇ρ|/ρ^{4/3} and t = τ/ρ^{5/3}.
  * **Open choices:**
    * the value of ε;
    * exactly how the opposite-spin "spin-averaged" s and t are formed;
    * whether the denominator basis includes the constant monomial.

    The paper's 11/12 term counts imply that it does.
  * A refit would be "the SAFS26-a form refit on our train with our L-BFGS". That is a fair *form* comparison (same fitter, same split), not their functional. With 50 parameters the fit is init/path-dependent, so use several restarts and report a range.
* **SAFS26-b (§2.5, Eqs. 15–17): not faithfully reproducible.** Unspecified:
  * which 11 terms are in the z-extended opposite-spin polynomial P_cos(w_avg, u_avg, z);
  * the values of the 5 same-spin polynomial coefficients "inherited and frozen from the parent";
  * the sigmoid constants a_x, b_x, a_ss, k_c usage and c_sig. These are not among the 19 trainable parameters, so they are fixed values that are never given.

  Any implementation would be our own variant.
* **Current policy:** compare with SAFS26 by **% change vs ωB97M-V** using their reported numbers (SAFS26-a −9.2 % val / −9.1 % tot; SAFS26-b −6.0 % / −5.9 %), because the absolute values come from a slightly different split (§5.1).
  * Option, not done yet: `baselines/safs26a.py` with the reconstructed form, refit non-SCF as an extra seed outside the iteration budget (minutes on one GPU, several restarts).

---

## 5. Dataset as implemented (`dftsr/dataset.py`)

* Source: ACCDB `Databases/MGCDB84/DatasetEval.csv` has **4986 lines**. The format is `name, coef, mol, coef, mol, ..., ref(Hartree)`.
* **XB51_15 dropped**, as in SyFES (its geometries are missing in the original SI) → **4985 reactions**.
* TAE140 / BDE99 / HAT707 are split into MR and nonMR subsets, as in SyFES: TAE140 indices 115, 121, 127–140 are MR, and any reaction containing those molecules is MR.
* **Splits (SyFES): train 870 / validation 2963 / test 1152** (paper: 870 / 2960 / 1150).
* Molecules: train 1329 (13,194 atoms), validation 3034 (30,088 atoms), test 1732 (31,325 atoms). Union used: **5930 molecules, 73,797 atoms** (MGCDB84_ALL.list has 5931).
* Weights: SyFES port. Subset weight 1/(count·RMS(ref)) is normalised within each group so the minimum is 1, then raised to the exponent ln2/ln(max) to land in [1, 2] (exponent 1 for RG10), then multiplied by the group weight. Positive-reference RG10 points are divided by 10⁴. **Validation weights are ×2.**
* WRMSD = sqrt(mean(w·(ΔE_kcal)²)), with 1 Hartree = 627.509474 kcal/mol.
* `load_reactions('agentic2026')` moves the **whole** C20C24 subset test → validation. This approximates the SAFS26 change (the paper moved "two molecules").
* **Geometry name escaping:** 89 MGCDB84 names do not match ACCDB file names. `=` becomes `_` (Bauza30: `23_Cl-_S=CF2_dim` → `23_Cl-_S_CF2_dim`), and `.` becomes `_` (SW49: `3.3.3-1_SW49` → `3_3_3-1_SW49`). `resolve_geometry()` tries the variants; all 5931 resolve.

Reaction counts by (type, split):

| type | train | validation | test |
|---|---|---|---|
| NCED | 311 | 1106 | 327 |
| NCEC | 20 | 120 | 103 |
| NCD | 21 | 69 | – |
| IE | 76 | 89 | 590 |
| ID | 67 | 80 | 8 |
| TCE | 287 | 600 | 60 |
| TCD | – | 234 | 24 |
| BH | 70 | 96 | 40 |
| AE18 | 18 | – | – |
| RG10 | – | 569 | – |

Mean weight per (type, split): AE18-train 1.0; BH 14.0 / 27.5 / 14.0; ID 13.3 / 30.8 / 16.3; IE 137.0 / 306.8 / 104.7; NCD 14.1 / 28.2; NCEC 130.0 / 221.2 / 116.1; NCED 139.5 / 235.9 / 117.0; RG10-val 17258.6; TCD 0.218 / 0.16; TCE 1.379 / 2.741 / 1.70. Order is train / val / test; validation includes the ×2 factor.

### 5.1 Split provenance and fidelity (recorded 2026-10-19)

**Origin.** The split is inherited from the ωB97M-V paper (Mardirossian & Head-Gordon, JCP 144, 214110 (2016)). There, train / validation / test are its "training set" / "primary test set" / "secondary test set". SyFES copied the split; SAFS26 kept it, with one change, so its numbers stay comparable.

**SyFES split: exact, not just the counts.**
* The split is defined **per subset**: each of the 84 subsets goes wholly to one split. Our `MGCDB84_INFO` in `dftsr/dataset.py` is a verbatim copy of `_MGCDB84_INFO` in the released SyFES code. The checked-out copy is at `<DATA_ROOT>/external/syfes_src/gr/symbolic_functionals/syfes/dataset/dataset.py`. The group weights (`_MGCDB84_WEIGHTS`) are copied the same way.
* We also apply the same `XB51_15` drop, the same MR/nonMR rule (TAE140 indices 115, 121, 127–140), the same weight algorithm, and the same `DatasetEval.csv` format. So membership is reaction-for-reaction identical.
* Check: ωB97M-V with published parameters gives train **3.319**, which matches [SAFS] 3.32 for the same train set.
* Count differences vs the papers (val 2963 vs "2960", test 1152 vs "1150"), probable causes:
  * the papers quote rounded or approximate numbers;
  * SyFES reports 7–8 unconverged molecules affecting 6–7 data points, whereas all 5930 of ours converged, so we keep every reaction.

  Not verified: which reactions SyFES actually lost is unknown.

| type | train | validation | test |
|---|---|---|---|
| NCED | A24, DS14, HB15, HSG, NBC10, S22, X40 | A21x12, BzDC215, HW30, NC15, S66, S66x8 | 3B-69-DIM, AlkBind12, CO2Nitrogen16, HB49, Ionic43 |
| NCEC | H2O6Bind8, HW6Cl, HW6F | FmH2O10, Shields38, SW49Bind345, SW49Bind6, WATER27 | 3B-69-TRIM, CE20, H2O20Bind10, H2O20Bind4 |
| NCD | TA13, XB18 | Bauza30, CT20, XB51 | – |
| IE | AlkIsomer11, Butanediol65 | ACONF, CYCONF, Pentane14, SW49Rel345, SW49Rel6 | H2O16Rel5, H2O20Rel10, H2O20Rel4, Melatonin52, YMPJ519 |
| ID | EIE22, Styrene45 | DIE60, ISOMERIZATION20 | C20C24 |
| TCE | AlkAtom19, BDE99nonMR, G21EA, G21IP, TAE140nonMR | AlkIsod14, BH76RC, EA13, HAT707nonMR, IP13, NBPRC, SN13 | BSR36, HNBrBDE18, WCPT6 |
| TCD | – | BDE99MR, HAT707MR, TAE140MR | PlatonicHD6, PlatonicID6, PlatonicIG6, PlatonicTAE6 |
| BH | BHPERI26, CRBH20, DBH24 | CR20, HTBH38, NHTBH38 | PX13, WCPT27 |
| AE18 | AE18 | – | – |
| RG10 | – | RG10 | – |

**SAFS26 split: approximate only.**
* [SAFS] moved "two C20C24 molecules" from test to validation without saying which. `agentic2026` moves the whole C20C24 subset.
* ωB97M-V on `agentic2026` is 3.319 / 4.086 / 3.632, tot 4.090 (n = 3546), vs [SAFS] 3.32 / 4.02 / 3.64, tot 4.07 (n = 3547). The 0.07 validation gap is larger than noise, so their split or setup differs slightly.
* Consequence: compare with SAFS26 by **% change vs ωB97M-V on our own split**, not by absolute WRMSD. The exact SAFS split would need the authors' file or the two molecule names.

**Why validation and test are larger than train.** These are the inferred design reasons of the ωB97M-V split:
1. A 12-parameter linear form needs only ~870 diverse points to fix its coefficients. The hard problem is choosing the form.
2. The "primary test" set was used to select among thousands of combinatorial term sets, so it had to be large and diverse. The same selection is why the published ωB97M-V parameters effectively saw train+validation, and why refitting on train alone worsens validation (3.53 → 3.87).
3. The "secondary test" set checks transfer to larger and harder systems: (H₂O)₂₀ clusters, C20C24, YMPJ519, Melatonin52, Platonic hydrocarbons. It is also the most expensive split (Σ nao³: test 2.69e12 vs val 0.96e12 vs train 0.41e12).

**Consequences for the search.** Because subsets are assigned whole, the splits have very different type mixes:
* train has no RG10 or TCD and few water clusters;
* validation is dominated by NCED (1106), TCE (600) and RG10 (569);
* test is dominated by IE (590 of 1152).

So flexibility that train cannot constrain overfits, and validation, which serves as the selection set, leaks over many iterations. Example: it159 beats GAS22 on validation but loses on test, mainly on IE (see `runs/scf_it159/comparison_table.md`).

Molecule statistics (all 5931):
* atoms per molecule: mean 12.44, median 11, max 60, total 73,802
* size histogram (atoms): 1: 198 · 2–4: 1675 · 5–9: 901 · 10–19: 1797 · 20–39: 1339 · 40–59: 7 · 60: 14
* largest systems: 60 atoms ((H2O)20 clusters, H2O20Bind10/4, H2O20Rel); 48 atoms ((H2O)16)
* multiplicities: singlet 5645, doublet 200, triplet 69, quartet 17
* elements (atom counts): H 38788, C 21568, O 6466, N 4092, F 663, S 367, Ar 334, Ne 329, He 312, Kr 269, Cl 228, Br 112, P 102, B 53, Si 48, Se 18, Al 12, Li 11, Be 10, Na 7, As 6, K 4, Mg 3. **Br, Kr, Se, As, K are beyond Ar**: SG-1 is unavailable for them.
* def2-QZVPPD basis functions: mean 595, median 513, max 2640. Percentiles 10/25/50/75/90/99 = 132/164/513/840/1290/1764. Σ nao³ = 4.05e12; by split, train 0.41e12, val 0.96e12, test 2.69e12. Most of the cost is in the test split.

---

## 6. Reference density campaign

### 6.1 Protocol (`scripts/compute_densities.py`)
1. `gpu4pyscf.dft.RKS/UKS(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')`, basis **def2-QZVPPD**, **grids.level = 3**, VV10 grid **SG-1 prune if max Z ≤ 18**, else the PySCF default nlcgrids. conv_tol 1e-10, max_cycle 100, starting from the default guess.
   * Fallback 1: level_shift 0.25 + damp 0.5, 300 cycles, from the previous dm.
   * Fallback 2: `.newton()`.
   * Unconverged molecules are still saved, with `converged=False`.
2. **Exact-integral (non-RI) energy components** at the converged RI density, using a non-DF KS object sharing the grids:
   * E_core = tr(h·D)
   * E_J = ½ tr(J·D)
   * E_xx = full-range HF exchange: −¼ tr(D·K) for RKS, −½ Σσ tr(Dσ·Kσ) for UKS
   * E_xx_lr = the same with `get_k(omega=0.3)`
   * E_sl = `nr_rks/nr_uks(..., 'wb97m-v')`
   * E_nlc = `nr_nlc_vxc`
   * E_nuc
   * **E_tot = E_core + E_J + E_nuc + E_sl + E_nlc + 0.15·E_xx + 0.85·E_xx_lr** (= 0.15 SR + 1.0 LR)
3. Grid features with gpu4pyscf `eval_ao(deriv=1, transpose=False)` + `eval_rho2(xctype='MGGA')`, in blocks of 32,768 points. Points with weight exactly 0 (padding) are removed. **gpu4pyscf grid weights can be negative** (e.g. min −29), so they are kept.
4. Check: libxc ωB97M-V on the stored features is compared with E_sl (`E_sl_grid_libxc`).
5. Write `<name>.h5.tmp`, then atomically rename to `<name>.h5`. Datasets (lzf-compressed):
   * `rho` float64 (nspin, 5, N) with rows [ρ, ∂xρ, ∂yρ, ∂zρ, τ] per spin. **RKS stores nspin = 1 with α values (= total/2).**
   * `weights` f64, `coords` f32, `mo_coeff`, `mo_occ`, `mo_energy`
   * attrs: E_core, E_J, E_nuc, E_sl, E_nlc, E_xx, E_xx_lr, E_tot, E_scf_df, E_sl_grid_libxc, converged, attempts, charge, mult, natm, nao, nelec, ngrid, basis, grid_level, xc, protocol, t_scf, t_exact, t_total, geometry.
6. Queue order: train molecules, then validation, then test, each sorted by size (smallest first). Workers coordinate via `_locks/<name>.lock` files (O_EXCL; stale locks with a dead pid are removed). Each result is logged as one JSON line to `_logs/<host>_gpu<N>.jsonl`.

Launch: `bash scripts/run_density_campaign.sh` (3 workers, OMP_NUM_THREADS = 16, `nohup setsid`, 20 s stagger). Stop: `pkill -f "python -u [s]cripts/compute_densities"`.

### 6.2 Why RI-JK SCF + exact energy (benchmarks, all ωB97M-V/def2-QZVPPD, grid level 3)

**Full non-RI SCF, GPU4PySCF, one GPU:**

| molecule | atoms | nao | grid pts | time | cycles | E (Ha) |
|---|---|---|---|---|---|---|
| water (A21x12 monA) | 3 | 132 | 33,792 | 20.6 s (incl. warm-up) | 8 | −76.43986492 |
| Al atom (W4-11) | 1 | 76 | 19,200 | 21.9 s | 9 | −242.35552010 |
| water dimer AB (3B-69) | 6 | 264 | 67,584 | 32.8 s | 8 | −152.87824701 |
| formaldehyde trimer (3B-69) | 12 | 585 | 143,616 | 110.2 s | 11 | −343.55180710 |
| acetic acid trimer (3B-69) | 24 | 1170 | 286,976 | 422.4 s | 12 | −687.37020971 |
| benzene trimer (3B-69) | 36 | 1728 | 430,848 | 1019.7 s | 10 | −696.73029580 |
| (H2O)20 isomer1 | 60 | 2640 | – | killed (not needed) | | |

**Full SCF, CPU PySCF, 32 OpenMP threads:** water 4.1 s (E −76.43986495); Al 2.7 s; water dimer 11.1 s; formaldehyde trimer 103.1 s; acetic acid trimer **1154.4 s** (vs 422 s on GPU). The GPU wins for large systems, and the CPU is no faster for small ones once GPU kernels are warm. The RTX 6000 Ada has weak FP64, which is why the GPU advantage is modest.

**RI-JK SCF + one exact-integral energy at the RI density, GPU:**

| molecule | nao | t_RI-SCF | t_total | E_RI | E_exact[ρ_RI] | E_exact[ρ_RI] − E_fullSCF |
|---|---|---|---|---|---|---|
| water | 132 | 3.0 s | 5.6 s | −76.43989995 | −76.43986480 | +1.2e-7 Ha |
| Al atom | 76 | 7.7 s | 10.2 s | −242.35553109 | −242.35552008 | +2e-8 |
| formaldehyde trimer | 585 | 6.8 s | 17.8 s | −343.55189894 | −343.55180628 | +8.2e-7 |
| acetic acid trimer | 1170 | 37.8 s | 80.3 s | −687.37041093 | −687.37020776 | +1.95e-6 (0.0012 kcal/mol) |
| benzene trimer | 1728 | 92.3 s | 225.3 s | −696.73054320 | −696.73029323 | +2.57e-6 (0.0016 kcal/mol) |
| (H2O)20 | 2640 | 341.4 / 340.1 s | 620.7 / 619.0 s | −1529.11808814 | −1529.11753867 | (no full reference) |

* The error is variational and second order, always positive and ≤ 3e-6 Ha. The raw RI energy error is 3.5e-5 to 5.5e-4 Ha, but that energy is not used.
* Speedup over full SCF: 6.2× (585), 5.3× (1170), 4.5× (1728). For large systems the exact-energy step (one exact J + K + K_lr) is ~55–60 % of the time.
* Production code timing on water: t_scf 1.6 s, t_exact 3.1 s, total 4.8 s. Al: 4.7 s. H atom: 0.8 s.

### 6.3 Campaign time estimate (method and numbers)
* Model: log-log interpolation of the RI+exact total time over (nao, t) = (33, 1.5), (132, 5.6), (585, 17.8), (1170, 80.3), (1728, 225.3), (2640, 620) s; ×1.10 + 2 s for features and IO; ×1.7 for open-shell (UKS).

| stage | new molecules | GPU-hours | wall on 3 GPUs | cumulative wall |
|---|---|---|---|---|
| train | 1329 | 10.0 | 3.3 h | 3.3 h |
| validation | 2869 | 22.3 | 7.4 h | 10.8 h |
| test | 1732 | 40.4 | 13.5 h | **24.2 h** |

* Total ≈ 73 GPU-hours. Equivalent full non-RI SCF ≈ 4–5× (4–5 days); CPU-only ≈ 10×.

### 6.4 Campaign status / history
* First launch ≈ 11:58. Restarted at ≈ 12:13 after the SG-1 bug (§12). Uncompleted molecules were redone; completed ones were kept.
* At ≈ 12:38: **train 544/1329 (40.9 %)**, validation 123/3034 (4.1 %), test 0/1732. 1.05 GPU-hours spent in 0.37 h wall. **0 errors, 0 unconverged**, all converged with plain DIIS. 1.4 GB on disk.
* Observed production timings so far: nao < 200: n = 465, mean 5.4 s (max 52.8 s), mean 32.5k grid points. nao 200–400: n = 79, mean 16.1 s (max 73.2 s), mean 60.3k points. Max |E_sl(libxc on stored features) − E_sl(numint)| = 7.4e-7 Ha.
* Grid points ≈ 11.3k–12k per atom (water 33.8k for 3 atoms; benzene trimer 431k for 36 atoms). Projected totals: **train ≈ 150M, validation ≈ 340M, test ≈ 360M points.**
* Disk projection: ~60 B/point for features + coords, plus MOs (Σ nao² × 8 B ≈ 20–30 GB). Total ~100–250 GB; the SSD has 1.9 TB.
* `post_density_pipeline.sh` was started at 12:15. It polls every 15 min (first log line: "missing: 3817"). It proceeds when train+val are complete **or** the density workers have exited (so a few failed molecules cannot block it). It then builds the caches, runs `establish_baseline.py` on GPU 0, waits for test, and builds the test cache. Env knobs: `ALLOW_MISSING` (default 0), `BASELINE_GPU` (default 0).

### 6.5 Constraint data (`scripts/compute_constraint_data.py`, CPU PySCF, done)
* **O2 triplet** (O–O 1.2075 Å), UKS ωB97M-V/def2-QZVPPD, level 3: E = −150.35821157 Ha, 28,168 points.
* **H2O** (A21x12 monA geometry): E = −76.43986495 Ha, 33,704 points.
* **AE18 atoms** (SCF on an unpruned (250,974) grid, features on (250,974) "fine" and (99,590) "coarse"). Energies (Ha):

| atom | E | atom | E | atom | E |
|---|---|---|---|---|---|
| H | −0.49459051 | He | −2.89865366 | Li | −7.47346901 |
| Be | −14.66233242 | B | −24.65125162 | C | −37.84824986 |
| N | −54.59500802 | O | −75.08099402 | F | −99.75045381 |
| Ne | −128.94920038 | Na | −162.26466443 | Mg | −200.06718255 |
| Al | −242.35552019 | Si | −289.36174083 | P | −341.25154043 |
| S | −398.10229827 | Cl | −460.13838836 | Ar | −527.52397577 |

  All converged.
* **Layout note:** CPU PySCF `eval_rho2(xctype='MGGA')` here returns **6 rows** [ρ, ∇ρ(3), ∇²ρ, τ], whereas gpu4pyscf returns **5 rows** [ρ, ∇ρ(3), τ]. Code that reads constraints.h5 takes **τ from the last row**.

---

## 7. Non-SCF evaluation design

### 7.1 Energy expression
E_mol(candidate) = E_tot^ωB97M-V + [E_HF-x(family) − E_HF-x(ωB97M-V)] + E_sl^cand[ρ] − E_sl,torch64^ωB97M-V[ρ]

* E_HF-x(family) = cx_sr·(E_xx − E_xx_lr) + cx_lr·E_xx_lr.
* Families (`api.FAMILIES`): **'rsh'** = ωB97M-V frame (ω 0.3, cx_sr 0.15, cx_lr 1.0, UEG g_x 0.85). **'mgga'** = no exact exchange (ω 0, UEG g_x 1.0). VV10 (b = 6, C = 0.01) is the same in both and fixed.
* The reference E_sl is recomputed in torch float64 (`E_sl_t64`, stored in the cache), so that the ωB97M-V candidate reproduces E_tot exactly. Difference energies keep float64 accumulation stable.

### 7.2 Feature caches (`dftsr/features.py`)
* Points with total ρ < **1e-11** are dropped.
* Closed-shell arrays: per-spin ρ, σ = |∇ρσ|², τ, w (float32), plus a mol index (int32).
* Open-shell arrays: ra, rb, saa, sab, sbb, ta, tb, w.
* Scalars (float64): E_tot, E_sl, E_sl_t64, E_xx, E_xx_lr, E_nlc, converged, npts.
* Built with `build_cache(split)`; loaded to the GPU by `SplitData`.
* **The Laplacian is not computed** by the GPU campaign (AOs with deriv = 1 only), and it is deliberately not exposed to candidates (SAFS26 anti-pattern: grid-sensitive and not scaling-invariant).

### 7.3 PyTorch XC (`dftsr/xc_torch.py`)
* LDA exchange per spin: ½·e_x^LDA(2ρσ).
* RSH attenuation F(a), with a = ω/k_Fσ and k_Fσ = (6π²ρσ)^{1/3}. An asymptotic series 1/(36a²) − 1/(960a⁴) + 1/(26880a⁶) is used for a ≥ 5 to avoid cancellation.
  **This series is wrong** (it is F(2a), the expansion for a' = ω/(2k_F)); the correct one is 1/(9a²) − 1/(60a⁴) + 1/(420a⁶).
  `DFTSR_RSH_SERIES=corrected` selects it; the default stays `legacy` so archived results reproduce. Impact: §13o (negligible).
* PW92 correlation with the **PW_MOD constants** (a = 0.0310907 / 0.01554535 / 0.0168869, fpp0 = 1.709920934161365617563962776245). These are libxc's choice for (ω)B97M-V. SyFES's plain PW92 constants give ~5e-6 Ha differences.
* Stoll partition into same-spin and opposite-spin parts.
* B97 descriptors: x² = σ/ρ^{8/3}; u = γx²/(1+γx²); t = τ_UEG/τ with τ_UEG = (3/10)(6π²)^{2/3}ρσ^{5/3}; w = (t−1)/(t+1).
* Opposite-spin averages: x²_avg = ½(x²_a + x²_b), t_avg = ½(t_a + t_b).
* Thresholds: RHO_THR 1e-10, TAU_THR 1e-14, SIGMA_THR 1e-20. Empty channels are masked. τ is clamped to ≥ τ_W.
* `Density.closed=True` makes channel b alias channel a (2× work saved). `scaled(λ)` gives ρ·λ³, σ·λ⁸, τ·λ⁵. `swapped()` exchanges the spin channels.

### 7.4 Candidate API (`dftsr/api.py`)
* A candidate is one Python file with exactly one `Functional` subclass: `family`, `init_params()` (**all** trainable parameters, all refit), `exchange(p, d)`, `correlation(p, d)`. Both return an energy density per volume [N].
* Helpers: `exchange_from_fx(d, fx, omega)`, `correlation_from_factors(d, fss, fos)`, `spin_sum`, `b97_u`, `b97_w`, `power_series`, `f_zeta`, `f_rsh`, `softplus`, `sigmoid`, `tanh`, `wigner_seitz_radius`.
* SpinChannel properties: rho, sigma, tau, x2, s2_pbe, t, w, tau_ueg, tau_w, alpha, rs, e_x_lda, e_c_lda_ss, mask.
* Density properties: a, b, rho, zeta, rs, sigma, tau, x2_avg, t_avg, w_avg, e_c_lda_decomposed().

### 7.5 Pipeline (`dftsr/worker.py: evaluate`)
1. Load the candidate. **Smoke test** on 4096 random closed-shell and 4096 open-shell points in float64 (ρ from 1e-6 to 1e2, fully polarised points, x up to 6, τ ≥ τ_W). Any non-finite value or wrong shape raises an error before GPU time is used.
2. **Fit** on train: loss = WRMSD_train². torch L-BFGS (strong Wolfe, history 30, tolerance_grad 1e-10, tolerance_change 1e-12), max_iter default 300, max_eval 1.5× that. Restarts after the first use init + N(0, 0.1·(|θ|+0.1)); the best training loss wins. Time limit 1800 s.
   * Two-pass gradient: pass 1 computes the per-molecule E (no grad) and dL/dE_mol; pass 2 recomputes per chunk and backpropagates Σ g[mol]·w·e.
   * Chunks of 2²² points, **always padded to the full chunk** (replicating the first point with zero weight), so each `torch.compile` graph has one fixed shape.
   * float32 kernels are compiled with `torch.compile(dynamic=False)`. If compilation fails, it **falls back automatically to eager** mode (the result reports `compiled`/`compile_fallback`).
   * A non-finite loss returns 1e8 and a zero gradient.
3. **Metrics in float64** (eager) on train and validation: WRMSD, per-type WRMSD and RMSD, per-subset RMSD, count of non-finite points, and missing-reaction counts.
   * Optional test metrics, plus "valtest_noRG10" (the SAFS "tot" metric).
4. **Constraints**, then **score** = J_target / val × 0.9^n_violations. Non-finite dataset points count as an extra violation.

### 7.6 Constraint tests (`dftsr/constraints.py`)

| test | definition | threshold |
|---|---|---|
| spin_symmetry | max over points with ρ > 1e-6 of \|e_xc(α,β) − e_xc(β,α)\|/ρ on the O2 triplet grid | 1e-5 Ha |
| ueg_exchange | g_x = e_x/Σσ e_x^LDA F_rsh at unpolarised UEG points r_s = 0.5, 1, 2, 5 (σ = 0, τ = τ_UEG); max \|g_x − 0.85\| (rsh) or \|g_x − 1\| (mgga) | 0.005 |
| x_scaling | H2O grid, λ = 0.5, 2, 5: Σ w·\|e_x^LDA-sr\|·\|g_x(ρ_λ) − g_x(ρ)\| over ρ > 1e-8, with the RSH attenuation divided out; worst λ | 1e-4 Ha |
| grid_ae18 | max over 18 atoms of \|E_sl(99,590) − E_sl(250,974)\|, same density | 0.015 kcal/mol |
| numerical | non-finite contributions on dataset points | 0 |

---

## 8. MCP server (`mcp_server/server.py`, name `dft-functional-search`)

* FastMCP (mcp 1.30), stdio transport. Originally registered in `.mcp.json` (project scope); **since 2026-10-15 it is registered at user (global) scope** (§13d).
* Env vars:
  * `DFTSR_GPUS` (default `"0"` in .mcp.json while densities run; set `"0,1,2"` afterwards)
  * `DFTSR_ALLOW_TEST` (`0`)
  * `DFTSR_JOB_TIMEOUT` (560 s since 2026-10-15; see §13d) and `DFTSR_FIT_BUDGET` (410 s)
  * `DFTSR_SCF_GATE` (0.97)
  * `DFTSR_RUNS_DIR`, `DFTSR_DATA_ROOT`, `DFTSR_DENSITY_TAG`
* **Worker pool:** one persistent process per GPU (spawn context). Each keeps the train and validation caches resident on its GPU. Workers start lazily on the first job, so the first evaluation also pays for loading the data. Workers are handed out through a queue of free workers. A timeout kills the worker; a crash restarts it on the next job.
* `evaluate_functional` and `final_test_evaluation` are async (`asyncio.to_thread`), so several evaluations can run in parallel, one per GPU.
* If Claude Code times out long MCP calls, start it with `MCP_TOOL_TIMEOUT=3600000 claude`.

| tool | purpose |
|---|---|
| get_task_brief | returns `prompts/NONSCF_SEARCH_PROMPT.md` + live status (baseline, J_target, caches, density count, GPUs, number of iterations) |
| evaluate_functional(name, code, plan, parent_id, island, n_restarts=1 [1–8], max_iter=300 [10–2000], tags) | syntax check → token delta → insert row (running) → write `candidates/<id>_<name>.py` → GPU evaluation → update row → mark `scf_status='eligible'` if 0 violations and val < 0.97 × baseline |
| get_iteration(id, include_code) | full record incl. per-type/per-subset errors and fit info |
| list_iterations(sort_by score/val_wrmsd/train_wrmsd/id/recent, limit, island, status, include_plans) | listing |
| get_lineage(id) | ancestors (≤ 50) with plans/summaries + children |
| record_summary(id, summary) | summarizer diagnosis (facts only) |
| add_memory(kind, text, iteration_ids) / get_memory(kind, limit) | global memory: dead_end, insight, todo, note |
| get_token_usage | cumulative project tokens + since last iteration + recorded totals |
| search_stats | number of iterations / ok / constraint-clean, best-so-far trajectory, tokens, GPU seconds |
| worker_status | pool state |
| scf_check_eligibility(id) | gate: status ok, 0 violations, val < 0.97 × ωB97M-V baseline |
| scf_submit(id, subset=validation/train/train+validation/all/probe, max_molecules, gpus) | detached `scripts/scf_evaluate.py` job; default GPUs = those not in DFTSR_GPUS |
| scf_status(job_id) | job row + progress.json |
| final_test_evaluation(id) | test metrics with the stored parameters (no refit); **disabled unless DFTSR_ALLOW_TEST=1** |

### 8.1 Database (`runs/iterations.sqlite`, `dftsr/tracker.py`)
* **iterations:** id, created, name, parent_id, island, family, status (running/ok/error), plan, code, code_path, n_params, params_json, train_wrmsd, val_wrmsd, test_wrmsd, score, n_violations, constraints_json, metrics_json, error, eval_seconds, **tokens_input, tokens_output, tokens_cache_read, tokens_cache_creation, tokens_total, cum_tokens_total**, session_id, summary, tags, scf_status.
* **memory:** id, created, kind, text, iteration_ids.
* **scf_jobs:** id, iteration_id, created, status, subset, result_json, log_path.
* **meta** (key/value): token_baseline, token_baseline_time, baseline_val_wrmsd, baseline_train_wrmsd, j_target, baseline_published_params.

### 8.2 Token accounting
* Source: all `*.jsonl` transcripts under `~/.claude/projects/-home-ash-Documents-dft-functional-sr/` (recursively, so subagent transcripts are included). Lines with `message.usage` are **de-duplicated by message id**, because a message with several content blocks repeats the same usage.
* Fields summed: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens. total = their sum. The most recent top-level transcript gives the session id.
* Per iteration: delta = current cumulative − stored baseline. The baseline is then advanced, so each iteration is charged with the tokens spent since the previous one (the planning/coding for it).
* The baseline is initialised at the first MCP server start (`ensure_token_baseline`), so setup work is excluded. The test DB created during setup was **deleted** so the real baseline starts with the user's next Claude session.
* For reference, this setup session at 12:19 had input 200, output 161,755, cache_read 19,722,038, cache_creation 305,484, total 20,189,477, over 100 unique messages. cache_read dominates the total; use the separate fields when reporting cost.

---

## 9. SCF evaluation (for later)

* `dftsr/scf.py: TorchNumInt` subclasses gpu4pyscf `NumInt` and overrides `eval_xc_eff`. The raw rho array ((5, N) total or (2, 5, N) per spin) goes cupy → torch via DLPack. Then e = energy_density, exc = e/ρ_total (0 where ρ < 1e-14), and **vxc = autograd ∂e/∂(ρ, ∇ρ, τ)** directly in GPU4PySCF's "eff" layout. Blocks of 2²⁰ points, float64. deriv ≤ 1 only, so SCF energies work but analytic Hessians/TDDFT do not.
* Hybrid and VV10 settings come from the frame xc name: 'rsh' → `wb97m-v`, 'mgga' → `b97m-v`. `make_ks()` builds an RI-JK KS object with the custom numint, grid level 3 and SG-1 (Z ≤ 18).
* `scripts/scf_evaluate.py`:
  * Subsets: probe (≤ 8 smallest validation reactions per type, ≈ 60 molecules), train, validation, test, train+validation, all. Options `--max-molecules` and `--available-only`.
  * One spawn process per GPU, with a shared queue.
  * Per molecule: the non-SCF energy from the stored features (same formula as the screening); an SCF warm-started from the stored ωB97M-V MOs (conv_tol 1e-9; fallback level shift/damping); then the exact-integral energy at the converged density.
  * Writes `molecules.jsonl` (resumable), `progress.json` (every 30 s) and `result.json` (SCF vs non-SCF WRMSD by type, density-driven error mean/max in kcal/mol, unconverged count). It updates `scf_jobs` and `iterations.scf_status`.
* **Validation tests:**
  * Custom-torch ωB97M-V SCF vs libxc ωB97M-V SCF (RI, def2-TZVP, level 3): H2O −76.4295946170 vs −76.4295946552 (Δ 3.8e-8 Ha; 2.3 s vs 1.7 s); OH radical (UKS) −75.7400562091 vs −75.7400562613 (Δ 5.2e-8; 2.1 s vs 2.1 s).
  * scf_evaluate on 6 AE18 atoms (H–C) with ωB97M-V parameters: E_scf − E_nonscf = 1.6e-8, 3.1e-8, 2.0e-7, 1.6e-7, 1.1e-7, 6.1e-8 Ha. WRMSD SCF 2.79370 vs non-SCF 2.79365. Per-atom times 0.5–7.1 s.
* **Time estimate for one SCF evaluation** (same cost model as the density campaign, warm start):

| subset | estimate |
|---|---|
| probe | ≈ 5–15 min |
| validation | ≈ 7–8 h on 3 GPUs (≈ 22 GPU-h) |
| train+validation | ≈ 11 h |
| all | ≈ 24 h |

  CPU-only would be ≈ 3–10× slower.

---

## 10. Measured numbers: non-SCF machinery

### 10.1 libxc ωB97M-V vs dftsr torch (def2-TZVP, level 3, UKS)

| system | libxc E_sl | torch64 (SyFES PW92) | torch64 (PW_MOD) | torch fp32 (PW_MOD) |
|---|---|---|---|---|
| H2O | −6.6644425033 | Δ −4.73e-6 | Δ +3.8e-8 | Δ −3.5e-7 |
| O2 triplet | −12.4398732682 | Δ −7.86e-6 | Δ +4.0e-8 | Δ −8.7e-7 |
| OH | −6.40615 | Δ −4.34e-6 | Δ +3.7e-8 | Δ −4.1e-7 |
| H | −0.1365450973 | Δ +3.3e-9 | Δ +6.4e-8 | Δ +8.6e-8 |

* On stored campaign densities (def2-QZVPPD), torch64 vs libxc E_sl: water 6.6e-8, Al 1.9e-7, H 1.6e-8 Ha.
* Energy-component closure (gpu4pyscf, def2-TZVP): RKS water 3.4e-12 Ha; UKS OH 2.8e-13 Ha (E_nlc 0.04259 / 0.03891; E_sl −6.66444 / −6.40615).

### 10.2 GPU throughput
* **Synthetic** (1e8 points, ωB97M-V form, forward + backward, chunk 4M, GPU 0 shared with a gpu4pyscf job):

| dtype / mode | compile time | per pass | per point |
|---|---|---|---|
| fp32 eager | – | 0.172 s | 1.72 ns |
| **fp32 compiled** | 14.9 s | **0.014 s** | **0.14 ns** |
| fp64 eager | – | 0.908 s | 9.08 ns |
| fp64 compiled | 14.3 s | 0.411 s | 4.11 ns |

* **Realistic size** (partial cache tiled to 150M train / 340M validation points, GPU 0 **shared with a running gpu4pyscf density worker at 100 % utilisation**):
  * ωB97M-V form: compile + first pass 2.7 s (inductor cache warm), **208 ms per objective+gradient evaluation** (46 evaluations in 9.5 s), fp64 train+val metrics 7.2 s.
  * GAS22 form: 8.8 s compile, **584 ms per evaluation** (49 in 28.6 s), metrics 9.2 s.
  * Peak GPU memory 17.4 GB.
  * The realistic per-evaluation cost is ~15× the synthetic one, because of GPU contention plus index_add/two-pass overhead. **Re-measure on a free GPU once the full caches exist.**
* Typical L-BFGS needs 100–200 evaluations (observed: 101–160 for ωB97M-V refits).
* **Estimate: ≈ 1–3 min per candidate per GPU, probably < 1 min on a free GPU.** 3 GPUs → 3 candidates in parallel. A 400-iteration search needs ≈ 7–20 GPU-hours of evaluation; LLM planning time per iteration (~1–3 min) is likely the bottleneck.

### 10.3 Literature numbers (WRMSD, kcal/mol; non-SCF on ωB97M-V densities; from [SAFS] Table 3)

| functional | train | val | test | tot (val+test, no RG10) |
|---|---|---|---|---|
| ωB97M-V | 3.32 | 4.02 | 3.64 | 4.07 |
| GAS22 | 3.27 | 3.74 | 3.19 | 3.77 |
| SAFS26-x (unconstrained) | 2.91 | 3.95 | 3.49 | 3.81 |
| SAFS26-a | 2.76 | 3.65 | 3.41 | 3.70 |
| SAFS26-b | 2.75 | 3.78 | 3.44 | 3.83 |
| AFS26-a (GPT 5.4) | 2.81 | 3.74 | 3.78 | 3.90 |
| AFS26-b (GPT 5.4) | 2.65 | 3.65 | 3.96 | 3.90 |

Per-category (val+test) for ωB97M-V / GAS22 / SAFS26-a / SAFS26-b:

| category | ωB97M-V | GAS22 | SAFS26-a | SAFS26-b |
|---|---|---|---|---|
| RG10 | 2.72 | 2.17 | 2.76 | 2.66 |
| NCED | 2.25 | 2.10 | 2.42 | 2.15 |
| NCEC | 6.35 | 5.81 | 5.96 | 5.95 |
| NCD | 2.36 | 2.37 | 2.79 | 2.75 |
| IE | 2.98 | 2.65 | 2.68 | 2.81 |
| ID | 13.25 | 10.93 | 10.30 | 12.23 |
| TCE | 3.97 | 4.16 | 3.63 | 3.78 |
| TCD | 1.99 | 2.36 | 1.76 | 2.56 |
| BH | 8.28 | 7.75 | 7.71 | 7.37 |

### 10.4 Pipeline test runs on PARTIAL data (not scientifically meaningful; plumbing checks only)
Data: 221 train molecules (95 of 870 reactions, mostly AE18 + TCE atoms), 53 validation molecules (116 TCE reactions); 5.2M + 1.3M points.

| run | train | val | notes |
|---|---|---|---|
| ωB97M-V published params (no fit) | 4.0225 | 4.4191 | 2.1 s. Constraints: spin 9.6e-16, UEG 4.6e-15 (g_x = 0.85 ×4), scaling 3.8e-15, grid 4.4e-5 kcal/mol (worst Mg); 0 violations. 13 params. |
| ωB97M-V refit, 2 restarts | 3.1308 (fp32 3.13081 vs fp64 3.13083) | 5.7569 | restart 0: 3.1308; restart 1: 3.2114; 160 evaluations; 32.7 s incl. compile. Overfits the tiny train set, as expected. |
| GAS22 published params | 3.894 | 4.611 | **UEG fails: 0.01214** (g_x = 0.8621); grid 1.59e-4. |
| B97M-V refit (mgga) | 3.606 | 5.493 | all constraints pass; 25.6 s |
| MCP end-to-end (stdio client) | 3.1743 | 5.7988 | 105 evaluations, 7.7 s. A deliberately broken candidate (`log(rho−1)`) was rejected by the smoke test (2925/4096 non-finite points). |

---

## 11. Documents and prompt

* `docs/FUNCTIONAL_FORMS.md` covers:
  * the common framework and notation mapping (B97 x ≡ SAFS s; t conventions);
  * B97 coefficients and the SyFES rediscovered form;
  * ωB97X-V;
  * B97M-V and ωB97M-V full coefficient tables;
  * GAS22 at full precision (from the colab);
  * SAFS26-a and SAFS26-b structures;
  * the AFS26 / ablation table;
  * the trajectory stepping stones (iterations 8 … 384);
  * the anti-patterns;
  * data/metric notes.
* `prompts/NONSCF_SEARCH_PROMPT.md` covers:
  * role and goal;
  * fixed vs designed parts;
  * the non-SCF caveat;
  * fitting/score/constraints;
  * the code API and rules (torch only, finite everywhere, exchange per spin, no Laplacian);
  * the protocol: session start → parent selection 80/20 with 4 islands A–D and migration every ~10 iterations → written plan → evaluate (≤ 2 retries) → factual summary → dead-end/insight memory every ~10 iterations → SCF gate (probe first, one job at a time);
  * parallelism;
  * integrity rules (no frozen-parameter grafting, no test use, no unphysical shortcuts, honesty, simplicity tie-break);
  * reporting every ~25 iterations, and a final top-5 report.
* SAFS26-a/b are **documented but not implemented** as runnable baselines, because the paper does not give their fitted parameters (details and the reconstruction option in §4.3).

---

## 12. Bugs, pitfalls and fixes encountered

1. **SG-1 VV10 prune undefined beyond Ar.** The Br atom (34_br_TA13) failed with `IndexError: index 35 is out of bounds for axis 0 with size 19`. Fix: use SG-1 only if max Z ≤ 18, otherwise the default nlcgrids. Workers were restarted ≈ 12:13; afterwards 0 errors.
2. **`pkill -f <pattern>` / `pgrep -f` killed the tool shell** when the pattern appeared in the shell's own command line (exit code 144, three times). Use the bracket trick: `pgrep -f "python -u [s]cripts/compute_densities"`.
3. **mcp 2.x** renamed FastMCP to MCPServer. Pinned `mcp<2`.
4. **PW92 constants:** libxc ωB97M-V uses PW_MOD. The SyFES constants give ~5e-6 Ha errors.
5. **τ row layout** differs between CPU PySCF (6 rows, Laplacian included) and gpu4pyscf (5 rows). The constraint loader uses the last row.
6. **Negative quadrature weights** in gpu4pyscf grids: do not filter on w > 0, only on w == 0 (padding).
7. **float32 and EPSILON = 1e-50** (SyFES) underflows to 0 in fp32. Explicit masks/clamps are used instead (RHO_THR 1e-10).
8. **exFAT 1 MB clusters:** a blob-less git checkout of ACCDB was extremely slow (lazy per-blob fetch). Re-cloned with `--depth 1` (45 s).
9. The **GPU4PySCF full SCF was slower than expected** (weak FP64 on Ada). This motivated the RI-JK + exact-energy protocol.
10. A **stdio MCP test with an explicit `env=`** does not pass `DFTSR_RUNS_DIR`. The test wrote 2 iterations into the real DB; the DB was deleted afterwards.
11. `claude mcp list` health checks start the server, which creates `runs/iterations.sqlite` and sets the token baseline. Harmless, but be aware of it.

---

## 13. Open items / TODO

* [ ] Wait for train+val densities (≈ 11 h from 12:13, i.e. ≈ 23:00 on 2026-10-13) and test (≈ 24 h, i.e. ≈ 12:00–13:00 on 2026-10-14). Check `post_density.log` and `density_progress.py`.
* [ ] Verify `establish_baseline.py`: does the ωB97M-V validation WRMSD (published params) reproduce ≈ 4.02 and train ≈ 3.32? If not, investigate (weights, MR split, grid, RI). J_target = 0.858 × baseline.
* [ ] Re-measure the non-SCF time per candidate on a free GPU with the full caches. Then set `DFTSR_GPUS="0,1,2"` in `.mcp.json` after the campaign.
* [ ] Handle molecules that fail or do not converge, if any (check `density_progress.py`; rerun with `--names`).
* [ ] Optional: runnable SAFS26-a/b reconstructions as seeds; an exact SAFS26 split variant (only 2 C20C24 molecules moved).
* [ ] Optional: a judge/penalty for frozen-parameter grafting (currently policy-only, via the prompt); periodic SCF refresh of densities.
* [ ] Later: extend to GSCDB137 (already cloned) to reduce validation leakage, as suggested in [SAFS] §4.
* [ ] Approve the MCP server in the next `claude` session in this directory; start the search with the prompt file.

---

## 13b. Status update 2026-10-14 12:30

* Densities: **train 1329/1329, validation 3034/3034, test 1619/1732 (93.5 %)**. 73.3 GPU-h used in 24.45 h wall, so the 24 h estimate held for train+val+most of test; the largest test molecules remain. **0 errors, 0 unconverged.** 65 GB on disk.
* Slowest molecules so far: 25_C22H10N2_CO2_IP_monA (nao 1842) 465 s; 26_C22H10N2_CO2_S_monA (1842) 452 s; 18_C24CageD3d (1512) 380 s.
* Train+val were complete at ≈ 00:04 (14 left at 00:04; the pipeline proceeded after the 00:19 check).
* Caches built at 00:19–00:21: **train 1329 molecules, 153.7M points; validation 3034 molecules, 348.7M points.** All 870 and 2963 reactions are present, 0 missing.
* **ωB97M-V with published parameters (non-SCF = its own SCF energies):**

| split variant | train | validation | test | paper |
|---|---|---|---|---|
| SyFES | **3.319** | **3.533** | 4.279 (1022/1152 reactions available) | [SAFS] train 3.32; [SyFES] test 4.237 |
| agentic2026 (all C20C24 moved to validation) | 3.319 | **4.086** | 3.498 (partial) | [SAFS] val 4.02, test 3.64 |

  * Train reproduces the paper exactly.
  * The [SAFS] validation value 4.02 is reproduced only when C20C24 is moved into validation. Its two large ωB97M-V errors (+12.5 and +13.0 kcal/mol, C20 cages vs bowl) dominate the change.
  * The stored baseline (meta) uses the SyFES split: baseline_val = 3.5336, **J_target = 3.0326** (= 0.858 × 3.5336).
  * Per-type validation WRMSD, published ωB97M-V: BH 8.311, ID 5.827, IE 2.259, NCD 2.357, NCEC 6.808, NCED 1.938, RG10 3.073, TCD 2.627, TCE 4.105.
  * C20C24_1 and C20C24_7 are trivial self-reactions (X − X, ref 0) in the original data; they are harmless.
* **Seed iterations** (refit on train, 2 restarts, GPU 0 shared with a density worker):

| seed | train | val | score | violations | time | evaluations |
|---|---|---|---|---|---|---|
| ωB97M-V refit | 3.184 | 3.870 | 0.784 | 0 | 104.5 s | 189 |
| GAS22 refit | 3.266 | **3.490** | 0.782 | 1 (UEG) | 54.9 s | 87 |
| B97M-V refit (mgga) | 6.159 | 6.276 | 0.483 | 0 | 106.7 s | 173 |

  * Refitting ωB97M-V on train only *worsens* validation (3.53 → 3.87). The published ωB97M-V parameters were selected using train+validation, so this is the expected overfitting signal.
* **Real non-SCF timing with full caches** (GPU shared with SCF): ≈ 0.3–0.55 s per objective+gradient evaluation; a full candidate with 2 restarts takes 55–107 s. **≈ 1–2 min per candidate** (1 restart is faster), consistent with the estimate.
* **Decision pending (user):** keep the SyFES split (baseline val 3.53) or switch to the agentic2026 split (val 4.09, closer to [SAFS]) before the search starts.

## 13c. Campaign complete (checked 2026-10-15 10:55)

* **All 5930 molecules done** (train 1329, validation 3034, test 1732). **0 errors, 0 unconverged** (all plain DIIS).
* **84.6 GPU-hours, 28.35 h wall** on 3 GPUs (estimate was 73 GPU-h / 24.2 h; the large test molecules were ~15 % slower than modelled). The last molecule finished ≈ 16:25 on 2026-10-14.
* Slowest molecule: 15_isomer5_H2O20Bind10, 789 s.
* Disk: densities 70 GB; feature caches 17 GB.
* Test cache built at 16:29 on 2026-10-14: **1732 molecules, 367.7M points.** `post_density_pipeline.sh` finished ("done").
* **Final ωB97M-V reproduction** (published parameters; non-SCF = SCF on its own density):

| split variant | train | val | test | tot (val+test, no RG10) | paper [SAFS] |
|---|---|---|---|---|---|
| **agentic2026** (C20C24 in validation) | **3.319** | **4.086** | **3.632** | **4.090** (n = 3546) | 3.32 / 4.02 / 3.64 / 4.07 (n = 3547) |
| SyFES | 3.319 | 3.533 | 4.309 | 3.867 | [SyFES] test 4.237 |

  The agentic2026 variant reproduces [SAFS] to within 0.07 kcal/mol in every column, which validates the densities, weights and splits.
* `.mcp.json` changed to **`DFTSR_GPUS="0,1,2"`**, since all GPUs are free. The change takes effect at the next Claude Code start.
* The baseline/J_target stored in the DB still use the SyFES split (J_target 3.0326). Switching to agentic2026 needs the worker's `split_variant` set and `establish_baseline.py` rerun (pending user decision).

## 13d. Global MCP server, 10-minute timeout, full tool test (2026-10-15 ~11:00–11:35)

### Registration (global)
* Removed the project-scoped entry; `.mcp.json` was deleted because it was left empty.
* Added at **user scope**: `claude mcp add -s user dft-functional-search -e DFTSR_GPUS=0,1,2 -e DFTSR_ALLOW_TEST=0 -e DFTSR_JOB_TIMEOUT=560 -e PYTHONWARNINGS=ignore -- .venv/bin/python mcp_server/server.py`. Stored in `~/.claude.json`.
* `claude mcp get` run from `/tmp` shows **"Scope: User config (available in all your projects) — ✔ Connected"**.
* **Real Claude Code check:** `claude -p` launched from `/tmp` called `search_stats` and `worker_status`, and returned n_iterations 3, baseline_val 3.5336 and gpus [0, 1, 2].
* Pre-warming: GPU data is loaded automatically only when Claude Code's cwd is inside the project, or when `DFTSR_PREWARM=1`. Elsewhere the workers stay lazy (no GPU memory used). New tool **`warm_up_workers(wait)`** (16 tools in total).

### Timeouts
* `~/.claude/settings.json` gets an `env` block: **`MCP_TOOL_TIMEOUT=600000`** (10 min per MCP tool call) and **`MCP_TIMEOUT=60000`** (server start-up). Existing keys were preserved.
* Server: **`DFTSR_JOB_TIMEOUT` default 560 s.** After that the worker is killed, a clean error is returned, and the worker restarts on the next call.
* **`DFTSR_FIT_BUDGET`** defaults to JOB_TIMEOUT − 150 = 410 s. When exhausted, the L-BFGS closure raises `FitTimeout`, and the best parameters seen so far (across restarts) are kept. The result reports `fit.timed_out` and `fit.fit_seconds`.
* `max_iter` is clamped to ≤ 1000 (was 2000), and restarts to ≤ 8.

### Token accounting changes
* `tracker.transcript_dirs()`: the project transcript dir, plus the server's launch-cwd dir if different, plus `DFTSR_TRANSCRIPT_DIRS`. Slug rule: every non-alphanumeric character becomes `-`.
* `runs/excluded_sessions.txt` lists session ids that are never counted. It currently holds **this setup session, 7d2c501e-ef9b-4396-be66-c407b740d55e**.
* The token baseline was reset to 0 at 2026-10-15 11:02:57, so the user's first search iteration is charged only for search work.

### Full tool test
Run against a **copy** of the DB (`DFTSR_RUNS_DIR=scratch`; the real DB is untouched: 3 seed iterations, 0 SCF jobs). **All 26 checks passed.** Results are saved in `runs/mcp_tooltest_results_2026-10-15.json`.

| check | time | result |
|---|---|---|
| initialize + list_tools | 0.3 s | 16 tools |
| warm_up_workers(wait) on 3 GPUs | **29.7 s** | train 153.7M pts loaded in 9.6 s, validation 348.7M in 17.6 s (page cache warm) |
| worker_status / get_task_brief | 0.0 s | ok; brief includes the new "Desirable outcomes" section |
| **3 × evaluate_functional in parallel** (free GPUs) | **52.5 s wall** | ωB97M-V refit **25.8 s** (66 evals; val 3.789, score 0.801, 0 violations). GAS22 refit **12.9 s** (18 evals; val 3.463, 1 violation). ωB97M-V + r_s-rational same-spin term (test form) **52.5 s** (69 evals; val 3.691, score 0.822, 0 violations) |
| evaluate_functional, broken code | 0.0 s | rejected by smoke test (2925/4096 non-finite) |
| evaluate_functional, syntax error | 0.0 s | rejected |
| get_iteration / list_iterations / get_lineage / record_summary / add_memory + get_memory / get_token_usage / search_stats | ≤ 0.1 s each | all correct |
| scf_check_eligibility | 0.0 s | eligible (clean, test gate 2.0); GAS22 refused (UEG violation) |
| scf_submit(probe, 4 molecules, GPU 0) + scf_status | **40 s** to done | 4/4 ok. SCF WRMSD 1.515 vs non-SCF 1.529 on the 4 probe reactions (refit ωB97M-V params). Density-driven error mean 0.095, max 0.18 kcal/mol |
| final_test_evaluation (DFTSR_ALLOW_TEST=1) | 7.9 s (test split loaded on demand) | refit ωB97M-V: train 3.191 / val 3.788 / **test 4.729** / val+test no RG10 4.134. Refitting on train worsens test vs published params (4.309) |
| final_test_evaluation (disabled) | 0.0 s | refused correctly |
| fit budget 8 s, 4 restarts | 14.2 s | `timed_out=true`, valid result (val 3.513) |
| job timeout 5 s | 5.0 s | clean error "evaluation timed out after 5s (worker restarted)" |
| next call after timeout | 8.4 s | worker restarted automatically |

* **Updated non-SCF time per candidate (free GPU): 13–52 s.** Three run in parallel on 3 GPUs, i.e. **~200–800 candidates per hour of GPU time**. LLM time per iteration will dominate.
* **SCF probe** of 4 small molecules: 40 s.
* Test artifacts were removed afterwards (`candidates/0000[4-10]_*_test.py` and the SSD dir `scf_runs/job0001_it00004`), so they cannot collide with real job or iteration ids.

### Prompt updates (`prompts/NONSCF_SEARCH_PROMPT.md`)
* Section 4.0: call `warm_up_workers` first, and read the latest results-checklist note.
* New time-limit paragraph: 10-minute cap, ~7-minute fit budget, ≤ 3 restarts, ≤ 500 iterations, simplify rather than retry on timeout.
* Parallelism: up to 3 parallel evaluations.
* **New Section 6, "Desirable outcomes (not required)",** from `improvement.md`:
  * a clean win over GAS22 and SAFS26-a with all 4 constraints;
  * surviving SCF;
  * interpretability (≤ 20 parameters);
  * method insight (train–val gap, improvement per token);
  * GSCDB137 transfer;
  * a margin table: GAS22 −7.0 % val / −7.4 % tot; SAFS26-a −9.2 % / −9.1 %; SAFS26-b −6.0 % / −5.9 % vs ωB97M-V;
  * an **11-row results checklist** to fill in each progress report and store via `add_memory(kind='note')`.
* Reporting renumbered to Section 7.

### Suggestions for good operation
1. **Decide the split before iteration 1:** agentic2026 (comparable to SAFS26, val 4.09) or SyFES (val 3.53). Do not change it mid-search.
2. **Pre-approve the MCP tools** so an autonomous search isn't interrupted by permission prompts: add `"mcp__dft-functional-search"` to `permissions.allow` in `~/.claude/settings.json` (user to decide).
3. **Run inside tmux/screen**, so a closed terminal does not kill the search. Resume with `claude --continue`; the DB and memory make context compaction safe.
4. **Version control:** `git init` + commit the evaluator code now, and tag the evaluator version used for each search. A changed evaluator invalidates comparisons between iterations.
5. **Back up `runs/iterations.sqlite`** regularly, e.g. a cron job copying it to the SSD every hour.
6. **Budgets / stopping:** set a token or iteration budget up front (e.g. 400 iterations, as in the paper). Stop early if the validation best has not improved in ~80 iterations. The paper saw validation leakage grow from 400 to 800 iterations.
7. **Islands via subagents:** run one subagent per island to keep the main context small; the main agent only migrates elites and synthesises dead ends.
8. **GPU 1 drives the main display.** If the desktop lags, set `DFTSR_GPUS=0,2`.
9. **Human checkpoints:** review the checklist every ~50 iterations. Run `final_test_evaluation` only at the end, on ≤ 5 candidates, with `DFTSR_ALLOW_TEST=1` set by hand.
10. **SCF later:** probe first (minutes), then validation (≈ 7–8 h on 3 GPUs). Pause the non-SCF search, or reduce `DFTSR_GPUS`, while a long SCF job runs.

## 13e. Per-iteration timing record (2026-10-15)

Every non-SCF calculation now records its timing. The values are returned by `evaluate_functional` and stored in `iterations`.

**New DB columns** (auto-migrated with `ALTER TABLE` in `tracker.connect()`; existing rows get NULL):

| column | meaning |
|---|---|
| `eval_seconds` | (existing) GPU evaluation time inside the worker |
| `wall_seconds` | wall time of the whole `evaluate_functional` tool call (submission → result) |
| `queue_seconds` | waiting for a free GPU worker, incl. worker cold start (data load) |
| `gap_seconds` | wall time since the previous non-seed iteration was submitted, i.e. the agent's planning/coding time |
| `timings_json` | per-stage breakdown (below) |

Stage keys in `timings_json`:
* `load_and_smoke_test`
* `compile_first_pass` (the torch.compile trigger)
* `fit`
* `fit_per_feval`
* `metrics_train_fp64`
* `metrics_validation_fp64`
* `metrics_test_fp64` (final test only)
* `constraints`
* `total_gpu_eval`
* `queue`
* `wall_tool_call`

Each stage is timed with `torch.cuda.synchronize()` so the GPU time is real.

**Surfacing:**
* `evaluate_functional` returns `timings_seconds`.
* `get_iteration` returns the columns + `timings_json`.
* `list_iterations` shows eval/wall/gap seconds.
* `search_stats` has a `timing` block (n / total / mean / median / max) for GPU eval, tool wall, queue, and agent gap between iterations.

The prompt asks for time in each progress report (checklist row 10b).

**Measured in the test** (test DB copy, GPU 0, worker started cold, from outside the project folder so no pre-warm):

| candidate | load+smoke | compile | fit | per eval | train metrics | val metrics | constraints | GPU total | queue | tool wall |
|---|---|---|---|---|---|---|---|---|---|---|
| ωB97M-V refit (cold worker) | 0.01 | 3.02 | 13.56 | 0.295 | 1.19 | 2.57 | 0.14 | **20.5** | 4.4 | 24.9 |
| GAS22 refit | 0.01 | 0.61 | 4.93 | 0.329 | 1.15 | 2.54 | 0.07 | **9.3** | 0.0 | 9.3 |

Agent gap between the two test calls: 28.2 s. Test files were removed afterwards; the real DB is migrated and still holds the 3 seeds.

## 13f. Campaign 2 finalists on the TEST split, non-SCF (2026-10-20)

Context: campaign 2 (iterations 201–400, 2026-10-19) is recorded in `runs/nonscf_continuation_2026-10-19/SUMMARY.md`
(robust-val shortlist: 396 robust 3.185, 377 3.231, 388 3.237, 374/376 3.294; no SCF yet). On 2026-10-20 the
user asked for a non-SCF evaluation of 388 and 376, then 396, **on the test split only**, compared with GAS22.
This is a human-authorized, reporting-only use of the test set: nothing was refitted, reselected or fed back.

* **Method:** stored fitted parameters, no refit, ωB97M-V/def2-QZVPPD densities, SyFES split, test only (1152
  reactions, 0 missing, 0 non-finite points). The MCP `final_test_evaluation` tool is disabled
  (`DFTSR_ALLOW_TEST` unset), so a standalone script calls the worker code directly (`Workspace(splits=('test',))`
  + `Evaluator.metrics`, `compile_=False`). Script `runs/nonscf_test_388_376/run_test.py [ids...]` (merges into
  `test_nonscf.json`); summary `runs/nonscf_test_388_376/RESULTS.md`.
* **Sanity checks:** it159 3.818 and published GAS22 3.5995 reproduce the earlier values (3.818 / 3.599) exactly.
* 376 = form-374 (ωB97M-V-ss branch + w-gated shell term) refit from the published-coefficient init.

| functional | params | val (non-SCF, stored) | **test WRMSD** | test without ID (n = 1144) |
|---|---|---|---|---|
| GAS22 (published) | 12 | 3.462 | **3.600** | **3.168** |
| it159 | 13 | 3.332 | 3.818 | 3.307 |
| it396 (path-fit params, a_s 2.78 / a_o −1.97) | 10 | **3.138** | 3.841 | 3.225 |
| it388 | 9 | 3.236 | 3.906 | 3.196 |
| it376 | 12 | 3.294 | 3.961 | 3.249 |
| ωB97M-V (published, SCF) | | 3.533 | 4.309 | |

Test WRMSD by type:

| type | n | it396 | it388 | it376 | it159 | GAS22 |
|---|---|---|---|---|---|---|
| BH | 40 | **5.90** | 6.06 | 7.23 | 6.58 | 6.88 |
| ID | 8 | 25.24 | 27.15 | 27.38 | 23.14 | **20.76** |
| IE | 590 | 3.03 | 2.93 | 2.81 | 3.15 | **2.73** |
| NCEC | 103 | **3.63** | 3.69 | 4.16 | 3.92 | 4.25 |
| NCED | 327 | 3.06 | 3.07 | 3.02 | 2.91 | **2.85** |
| TCD | 24 | 2.78 | 2.84 | 2.91 | **2.67** | 3.95 |
| TCE | 60 | 2.87 | 2.81 | 2.54 | 2.55 | **2.22** |

* **Conclusions:**
  * The campaign-2 validation gains do not transfer to test. GAS22 still has the lowest test WRMSD.
  * 396 is the best new form on test (3.841) but is behind it159 (3.818). Its 0.10 val lead over 388 shrinks to
    0.065 on test.
  * The new forms win BH, NCEC and TCD (vs GAS22) and lose ID, IE, NCED and TCE.
  * Within TCE the gap is mainly BSR36 (subset RMSD: 1.72 / 1.64 for 396 / 388 vs 0.58 for GAS22) and
    PlatonicTAE6.
  * This fits the SUMMARY.md warning that the shell-term val numbers are optimistic: the shell region was found
    with a train+val residual scan.
  * Only one parameter set of 396 was tested; its published/distant-init sets have different shell amplitudes.
* **"Test without ID"** is a post-hoc diagnostic, not a headline metric. ID on test is only C20C24, 8 reactions with
  21–27 kcal/mol errors for every functional. Those 8 reactions are 23–33 % of the squared error. The value is
  recomputed as sqrt((Σ_types n·WRMSD² − 8·WRMSD_ID²) / (1152 − 8)), in the same way as val+test-without-RG10.

## 13g. GSCDB137 SCF transfer test: setup (2026-10-20)

Next task (user, 2026-10-20): run the actual SCF of the **iteration-159 functional** on GSCDB137 (Liang &
Head-Gordon, JCTC 21, 12601 (2025)). The work happens in **`/home/ash/Documents/dft_fucntional_sr_gold_std`**
(folder name spelled as the user created it; it holds the paper `ct5c01380.pdf` + SI). The code was added to this
repo's MCP server **without changing existing code** (the user will run it from a new Claude session in that
folder). **No it159 GSCDB SCF job has been launched; it needs the user's explicit approval** (tool gate
`user_approval="it159"`). Full operating notes for that session: `dft_fucntional_sr_gold_std/GSCDB_NOTES.md`.

* **Dataset:** `git clone --depth 1 https://github.com/JiashuLiang/GSCDB` → `dft_fucntional_sr_gold_std/GSCDB/`
  (v1.0.0, 268 MB). This replaces the older 46 GB `GSCDB138` clone listed in §3.2. The analysis uses 137 sets,
  8377 entries and 13,907 molecules (SC74/OEEFD excluded as in `Analysis/analyze.ipynb`). Q-Chem inputs are the
  authority. `Analysis/Molecule_Energies.csv` has GSCDB's own Q-Chem energies of 16 functionals for every
  molecule, including ωB97M-V, which serves as an exact check of our setup.
* **New dependency:** `basis_set_exchange` (installed into `.venv` with `uv pip`; only needed for AUG-CC-PCV5Z, 14
  AE18 atoms).
* **New code (additions only):**
  * `dftsr/gscdb.py`:
    * Q-Chem input parser: geometry, charge/multiplicity, named or `$basis` basis, `$ecp`, `$multipole_field`,
      XC_GRID, NL_GRID, SCF_GUESS_MIX, INTERNAL_STABILITY.
    * PySCF molecule builder: named basis via PySCF, else BSE; element blocks that just name a basis (DAPD);
      core-valence sets fall back to valence sets for H/He; def2 ECPs for Z > 36; Gaussian-format ECP converter.
    * Uniform-field term.
    * Dataset tables.
    * Port of `analyze.ipynb`.
  * `dftsr/gscdb_scf.py`: SCF protocol.
  * `dftsr/gscdb_jobs.py`: job registry (`runs/gscdb_jobs.sqlite` in the GSCDB folder) and reports.
  * `scripts/gscdb_scf.py`: multi-GPU runner, resumable, with progress/ETA, `result.json` and `energies.csv` in
    Molecule_Energies layout.
  * MCP tools appended to `mcp_server/server.py`: `gscdb_info`, `gscdb_scf_submit` (approval-gated, launches a
    `systemd-run --user` unit `gscdb-jobNNNN`), `gscdb_scf_status`, `gscdb_report` (works on partial runs),
    `gscdb_stop`, `gscdb_resume`.
* **Protocol per molecule:**
  1. ωB97M-V (libxc) SCF in the GSCDB setup. RI-JK + exact-energy only for def2 basis sets, exact JK otherwise.
     State search if the energy is > 1e-4 Eh above GSCDB's published ωB97M-V energy. Orbitals stored in
     `/media/ash/PortableSSD/dft_sr/gscdb_ref_wb97mv/` (≈ 93 GB when complete) and reused by later candidates.
  2. Candidate non-SCF on that density.
  3. Candidate SCF from those orbitals.

  Metric: GSCDB mean NER vs the 16 published functionals, over the datasets complete for every method.
* **Verified so far:**
  * The analysis port reproduces the published mean NER of all 16 functionals to ≤ 2e-9 (137 sets).
  * All 13,907 molecules build. Basis-function counts equal Q-Chem's `num_basis` for 13,900. The 7 O24x5 He₂
    systems differ (105 vs 33 functions per He), but the JSON count is just stale: with exact JK our ωB97M-V
    energies match Q-Chem to ≤ 1e-7 Eh. The 6.7e-4 Eh gap in the first probe was DF error.
  * Q-Chem field convention: H = H0 − F·(Σ r_i − Σ Z_A R_A), origin (0,0,0). With this sign the Dip146 AlF
    pair matches the published energies with the same 2.75e-6 Eh offset as the field-free molecule.
  * **Validation probe v2** (`dft_fucntional_sr_gold_std/runs/validation_probe_v2_wb97mv_torch/`): 42 molecules,
    the 1–2 smallest of every special case; candidate = the torch ωB97M-V `baselines/wb97mv.py`.
    * 42/42 ok, 0 errors, 0 unconverged.
    * Our libxc ωB97M-V − published Q-Chem ωB97M-V: median |Δ| 9e-5 kcal/mol, mean 0.009, max 0.117 kcal/mol.
      The worst cases are open-shell TM atoms (3d4dIPSS Co⁺ excited state 0.117, Sc⁺/Sc 0.078, V excited state
      0.045), the Au atom 0.029 and C60ISO_1 0.012. Everything else is ≤ 0.004 kcal/mol, typically 1e-7 Eh.
    * Torch-ωB97M-V SCF / non-SCF minus libxc ωB97M-V: max 1.3e-5 kcal/mol. This validates the candidate path
      (TorchNumInt, fields, ECPs incl. the CPU core Hamiltonian, h functions, broken-symmetry UKS).
    * The state search fixed ALKBDE10_mgo (+2.3e-3 Eh → 3.7e-6 Eh *below* Q-Chem) and Sc⁺ (+1.0e-2 → +1.2e-4 Eh).
    * Measured costs (ref + candidate, one GPU): small molecules 1–15 s; aniline dimer (1344, DF) 455 s;
      C60ISO (2220, host-memory DF) 3170 s; exact-JK 3d4dIPSS atoms 70–480 s.
  * **End-to-end check on complete datasets** (`runs/validation_sets_wb97mv_torch/`: AE18, TMD10, DBH22, TA13, SN13,
WCPT6, ALKBDE10; 163 molecules, 36 min on 3 GPUs, 0 errors, 0 unconverged). Our ωB97M-V − published Q-Chem
ωB97M-V: median 6e-4, max 0.053 kcal/mol (TMD08_V). The per-set metrics reproduce GSCDB's published ωB97M-V
metrics to ~0.1 % (e.g. AE18 5.3082 vs 5.3085, TMD10 6.4582 vs 6.4673, TA13 1.6664 vs 1.6663). Mean NER over
these 7 sets: 0.976 (ours) vs 0.977 (published). The torch-ωB97M-V candidate gives identical numbers, SCF and
non-SCF.
* **Cost estimate** (`gscdb_estimate`; model t = 4 s · atoms · grid-size factor + B · (nao/1344)^2.7, with
  B = 351 s DF or 1300 s exact JK, × 2 for host-memory DF):
  * max_nbas 2000: ≈ 630 GPU-h (13,489 molecules).
  * max_nbas 3000: ≈ 960 (model) – 1590 (calibrated on the validation runs) GPU-h, i.e. 13–22 days on 3 GPUs
    (13,806 molecules; 131/137 sets complete).
  * All molecules: ≈ 1600 GPU-h.
  * The 101 molecules above 3000 functions (MME52, UPU23, ROST61, BH876, ISOL23, MOR13) would need exact JK
    (RI tensor too big for host memory), ≈ 650 GPU-h. Treat all of these as ±50 %.
  * Cheaper options not yet taken: a better RI fitting basis for aug-pc-n (e.g. PySCF auto/etb aux) so the
    1077 exact-JK aug-pc-3 property molecules could use DF; pruned grids (would deviate from GSCDB's protocol).
* **Problems found and fixed:**
  1. Upper-case basis names (`DEF2-QZVPPD`) and AE11 `$basis` headers without "0".
  2. GPU4PySCF's ECP kernel fails for l = 5 shells (`ECP_cart (5,5,3)`, P34 aug-cc-pwCVQZ-PP): the core
     Hamiltonian is built on the CPU for ECP molecules with h functions.
  3. cuSOLVER generalized `eigh` fails for some broken-symmetry guesses: the guess diagonalisation is done on
     the CPU (SciPy).
  4. RI-JK with def2-universal-jkfit is not accurate for non-def2 basis sets (EA50 Li with aug-pc-4: DF energy
     1e-3 Eh off, exact energy at the DF density 1.7e-4 Eh off): DF is restricted to def2 basis sets.
  5. Open-shell TM atoms converge to higher states than Q-Chem found (3d4dIPSS Sc⁺ +1.0e-2 Eh, Co⁺ excited
     state +1.4e-2 Eh). Added the state search: other guesses, then CPU PySCF stability analysis +
     instability following (GPU4PySCF has no UKS stability for this functional). On Sc⁺ it went
     1.0e-2 → 9.9e-4 → 1.2e-4 Eh in 3 rounds of ~4 min.
  6. **systemd-oomd killed the probe unit twice** (user-slice memory pressure > 50 %, 17:03 and 18:40). Two
     C60ISO DF calculations ran at once, and GPU4PySCF keeps RI tensors that do not fit on the GPU in pinned
     host memory (72 GB for one C60ISO; the 1344-function aniline dimer stays on the GPU). Fix: an estimated
     host-memory size (2.2 · nao · n_pairs · 8 B − 25 GB) reserved from a shared budget
     (`GSCDB_HOST_BUDGET_GB`, default 75, i.e. one C60ISO at a time; the measured pressure was then ≤ 0.6 %).
     Molecules whose tensor exceeds the budget use exact JK. User units cannot opt out of oomd
     (ManagedOOMPreference is ignored for non-root cgroups).

## 13h. GSCDB137 it159 on the 36 sets ≤ 500 basis functions: results (job 0001, 2026-10-20 → 10-23)

Report: `/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0001_it159_datasets/REPORT.md` (notes:
`GSCDB_NOTES.md` there). 3,486 of 3,487 molecules, 0 errors, 48.1 h unit time, 138.2 GPU-h (20.1 of them in
records invalidated by mid-run fixes). AE11_Rn excluded (GPU XC defect for all-electron Rn) → 35 sets compared.
- Mean NER (35 sets): ωB97M2 0.930, ωB97M-V publ. 1.077, ωB97M-V ours 1.086, **it159 SCF 1.099 (rank 3 of the 16
  published)**, it159 non-SCF 1.101, ωB97X-V 1.197, CF22D 1.233. 19 new sets: it159 1.202, rank 4 (behind ωB97M2
  0.953, ωB97X-V 1.155, ωB97M-V 1.186). 16 overlapping sets: rank 3. MGCDB84 train 0.981 (rank 2), val 1.004, test 0.676.
- it159 − ωB97M-V (ours): ΔNER +0.013 (35 sets; Wilcoxon p 0.39; bootstrap CI [−0.018, +0.043]), +0.011 on the new
  sets (p 0.47): indistinguishable. Better on noncovalent, worse on TM / field properties / V30.
- Density-driven error mean 0.038, max 1.29 kcal/mol (W4-17_p4); 15 it159 SCF unconverged (no Newton for the torch
  functional), all ≤ 0.12 kcal/mol from non-SCF.
- Our ωB97M-V vs Q-Chem: median 3.6e-4 kcal/mol; 4 molecules > 1 kcal/mol, all state differences; on the 29
  unaffected sets our mean NER = published (1.077 vs 1.078).
- Fixes in `dftsr/gscdb_scf.py` + `scripts/gscdb_scf.py` (GSCDB-only; MGCDB84 code untouched): three GPU4PySCF 1.8.1
  h-shell defects (point-charge nuclear attraction → CPU hcore for l ≥ 5; fused range-separated K → split K;
  semilocal XC for Rn exponent 2.3e8 → not fixable, excluded), VV10 SG-1 guard with true Z, CPU stability with a
  CuPy hcore, stability re-SCF fallbacks, field-sibling guess in the state search, `redo_refs.txt` in the runner.
  All 467 h-shell molecules were recomputed. The MGCDB84 SCF path (def2-QZVPPD, lmax 4) does not use h shells.

## 13i. GSCDB it396 non-SCF on the same 36 sets (job 0002, 2026-10-23)

Report: `/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0002_it396_datasets/REPORT.md` (pre-registered prompt
in the same folder). The user picked it396 on MGCDB84 criteria (best val 3.138). New runner mode `--nonscf-only` or
a `NONSCF_ONLY` file (`scripts/gscdb_scf.py`, `dftsr/gscdb_scf.py::candidate_nonscf`, `dftsr/gscdb_jobs.py::finalize`;
GSCDB-only). It reproduces job 0001 non-SCF to 9e-12 kcal/mol and reuses the stored ωB97M-V orbitals:
7.4 GPU-h, 2.5 h wall, 0 errors.
- 35 sets: it396 non-SCF 1.110 vs it159 non-SCF 1.101 (SCF 1.099); rank 3 of the 16 published; ΔNER +0.009, p 0.78.
- 19 new sets: 1.258 vs 1.204, significantly worse (p 0.018, CI [+0.007, +0.116]; DAPD +0.50).
  16 MGCDB84-derived sets: 0.935 vs 0.979 (rank 2).
- MGCDB84 val gains do not transfer; this agrees with the MGCDB84 test split (3.841 vs 3.818). No it396 SCF suggested.
- Note: the running MCP server keeps the old `finalize` until it restarts, so `gscdb_report` shows only the ωB97M-V
  table for non-SCF-only jobs until then.

## 13j. GSCDB GAS22 non-SCF on the same 36 sets (job 0003, 2026-10-23)

Report: `/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0003_GAS22_datasets/REPORT.md`. GAS22 (published
parameters, `baselines/gas22.py`) is not an MCP iteration, so it was registered as job 3 (iteration_id 0) and launched
with systemd-run in the `gscdb_scf_submit` layout. Non-SCF mode, 7.3 GPU-h, 0 errors, AE11_Rn excluded.
- 35 sets: GAS22 1.228 (rank 4 of 16), significantly worse than it159 non-SCF 1.101 (p 0.005), it396 1.110
  (p 0.023) and our ωB97M-V 1.086 (p 0.003). Worst on transition metals (2.20) and thermochemistry (1.20); best of
  our three only on noncovalent sets (0.861).
- This reverses the MGCDB84 test order (GAS22 3.600 best). GSCDB weights TM, heavy elements and properties, which
  MGCDB84 lacks.
- Order on the subset: ωB97M-V ours 1.086 < it159 1.099 (SCF) / 1.101 < it396 1.110 < GAS22 1.228.

## 13j-2. GSCDB GAS22 SCF on the same 36 sets (job 0004, 2026-10-23/24)

Report: `/home/ash/Documents/dft_fucntional_sr_gold_std/runs/job0004_GAS22_SCF_datasets/REPORT.md`. GAS22 (published
parameters), SCF + non-SCF from the stored ωB97M-V orbitals, registered as job 4 (iteration_id 0), systemd-run unit
`gscdb-job0004`. 65.3 GPU-h (estimate 50.4), 21.8 h wall, 3,486 molecules, 0 errors, AE11_Rn pre-excluded.
- Gate: references identical to job 0001 (E_ref exact), non-SCF identical to job 0003 (1.6e-9 kcal/mol). The
  unconverged-candidate rule (≤ 1 %) failed (20/303 at the gate, 66/3,486 = 1.9 % at the end). The job was stopped,
  reported and resumed on the user's go-ahead. All 66 hit DIIS 150 + level shift 300, then `NotImplementedError` (no Newton
  for torch functionals); they sit within 0.84 kcal/mol of non-SCF. it159 had 15.
- 35 sets: GAS22 SCF **1.205** (rank 4 of 16 published), non-SCF 1.227 (job 0003's "1.228" was a rounding slip).
  SCF improves GAS22 by 0.022 NER (p 0.0003; 27/35 sets), unlike it159 (0.002). Density-driven error mean 0.150,
  max 1.92 kcal/mol (it159 0.038 / 1.29).
- Still significantly worse than it159 SCF (+0.106, p 0.006; new sets +0.151, p 0.016) and ωB97M-V (+0.119, p 0.008).
  Wins no set. On the 19 new sets 1.353 (rank 6); on the 16 MGCDB84 sets 1.029 (rank 3).
- Order on the subset: ωB97M-V ours 1.086 < it159 1.099 < it396 1.110 < GAS22 SCF 1.205.
- **AE11_Rn (2026-10-24):** computed afterwards for this job with a standalone CPU script
  (`runs/job0004_GAS22_SCF_datasets/ae11_cpu_fallback.py`, no project code changed; validated on AE11_Kr to 5e-6
  kcal/mol; ωB97M-V − Q-Chem +0.003 kcal/mol, 61 s). Job 0004 has all 3,487 molecules. Stored `AE11_Rn.h5` replaced (old
  one backed up). Over 36 sets GAS22 SCF 1.190 (rank 3 of 16) vs ωB97M-V ours 1.144; AE11 NER 0.67 vs 3.15. it159/it396
  lack AE11_Rn at that point.
- **AE11_Rn for every job (2026-10-24, user approval):** added to jobs 0001–0003 with the same script (`--ref-from-store`,
  `--nonscf-only`); all four jobs have 3,487 molecules. **36 sets:** it159 SCF 1.139 (rank 3 of 16; rank 2 on the 20 new sets) < ωB97M-V
  ours 1.144 < it396 1.147 < GAS22 SCF 1.190. AE11 NER: it159 2.53, it396 2.43, GAS22 0.67, ωB97M-V 3.15. it159 vs ωB97M-V −0.005
  (p 0.58); GAS22 vs it159 +0.052, Wilcoxon p 0.020 but bootstrap CI [−0.085, +0.165]: not robust. On 35 sets (no AE11) the
  gap was +0.106 (p 0.006). Tables: job 0004 REPORT.md §11.
- **Convergence fix (2026-10-24):** unconverged SCFs of the three functionals with an SCF made to converge (`runs/convergence_fix/README.md` in the GSCDB
  folder, no project code changed). Cause: GAS22/it159 torch potentials are non-smooth in the far minority-spin tail (ρσ ≈ 1e-10); libxc ωB97M-V and torch-ωB97M-V
  are not. ωB97M-V refs 2 → 0 (CPU, stored orbitals replaced), it159 15 → 0, GAS22 66 → 2 (the Pd atom, twice). 12 GAS22 and 3 it159 records strictly
  converged (|g| < 1e-6), 52 + 12 noise-limited (best iterate |g| < 1e-3, 5 best energies within ≤ 8.3e-8 Eh). Energies moved ≤ 3.9e-4 kcal/mol, mean
  NER ≤ 1e-5, no ranking or test changed.

## 13k. WRMSD_tot of the SCF runs on the paper's split (2026-10-23)

The user asked for WRMSD_tot of every functional with a full MGCDB84 SCF run, on the split SAFS26
([SAFS] = `agentic_discovery_paper_2026.pdf`) uses for its non-SCF numbers.
- **Split:** `load_reactions('agentic2026')`, i.e. SyFES with the whole C20C24 subset moved test → validation (§5.1;
  the paper moved "two molecules", so the split is approximate). This changes the weights: validation weights are ×2.
- **WRMSD_tot:** validation + test without RG10, n = 3546 (paper 3547), as in [SAFS] Table 1/3.
- **Sources:** per-molecule energies in `molecules.jsonl` of `/media/ash/PortableSSD/dft_sr/scf_runs/job0002_it00159`
  and `.../gas22_published`. ωB97M-V = the stored self-consistent E_tot. No refit, no new SCF.
- **Script / output:** `runs/scf_wrmsd_tot_agentic2026/wrmsd_tot.py` → `wrmsd_tot.json` (both splits).
- **Check:** on the SyFES split it reproduces the stored values exactly (it159 SCF 3.253 / 3.330 / 3.805 / 3.607;
  GAS22 SCF 3.264 / 3.470 / 3.589 / 3.664). ωB97M-V on agentic2026 gives 3.319 / 4.086 / 3.632 / 4.090, the same as §5.1.
- Only it159 and GAS22 have full MGCDB84 SCF runs. The campaign-2 forms (it396, it388, it376, …) have none.

| functional | train | val | test | **WRMSD_tot** | vs ωB97M-V | [SAFS] tot (non-SCF) |
|---|---|---|---|---|---|---|
| ωB97M-V (SCF reference) | 3.319 | 4.086 | 3.632 | **4.090** | — | 4.07 |
| GAS22 SCF (published params) | 3.264 | 3.782 | 3.161 | **3.793** | −7.3 % | 3.77 |
| GAS22 non-SCF | 3.270 | 3.778 | 3.168 | 3.788 | −7.4 % | |
| it159 SCF | 3.253 | 3.726 | 3.304 | **3.767** | −7.9 % | |
| it159 non-SCF | 3.252 | 3.736 | 3.307 | 3.776 | −7.7 % | |
| SAFS26-a / SAFS26-b ([SAFS], non-SCF) | 2.76 / 2.75 | 3.65 / 3.78 | 3.41 / 3.44 | 3.70 / 3.83 | −9.1 % / −5.9 % | |

n: train 870, val 2971, test 1144, tot 3546. All values kcal/mol.

WRMSD_tot by category (ωB97M-V / GAS22 SCF / it159 SCF):

| category | n | ωB97M-V | GAS22 SCF | it159 SCF |
|---|---|---|---|---|
| BH | 136 | 8.28 | 7.83 | **7.31** |
| ID | 88 | 13.20 | **10.78** | 11.42 |
| IE | 679 | 2.97 | **2.61** | 3.02 |
| NCD | 69 | **2.36** | 2.37 | 2.42 |
| NCEC | 223 | 6.34 | 5.80 | **5.36** |
| NCED | 1433 | 2.25 | **2.11** | 2.16 |
| TCD | 258 | **2.56** | 2.81 | 2.65 |
| TCE | 660 | 3.97 | 4.17 | **3.90** |

- it159 SCF has the lowest WRMSD_tot (3.767), 0.026 below GAS22 SCF (3.793). This is the same order as the SyFES
  val+test-no-RG10 metric (3.607 vs 3.664), but the gap shrinks. Moving C20C24 (ID, where GAS22 is better) into
  validation doubles its weight.
- GAS22 still has the lower test WRMSD on this split (3.161 vs it159 3.304), so the test-split order is unchanged.
- SCF vs non-SCF changes WRMSD_tot by < 0.01 for both functionals.
- Our ωB97M-V and GAS22 per-category numbers match [SAFS] to ≤ 0.05 except TCD (ours 2.56 / 2.81 vs their 1.99 / 2.36).
  Not investigated.

**Addendum (2026-10-31): why [SAFS] reports GAS22 test 3.19 while §13j has 3.600**
- The two numbers use different splits, not different results. §13j uses the SyFES split, where the test split contains
  C20C24: 8 ID reactions on which every functional has errors above 20 kcal/mol (GAS22 20.8, it159 23.1). [SAFS] moves
  C20C24 to validation. Without it, the test split is 1144 reactions, and GAS22 test is 3.168 non-SCF / 3.161 SCF.
  Compare the §13j column "test without ID (n = 1144)": 3.168.
- So for comparisons with the [SAFS] tables, use the agentic2026 numbers in this section, not §13j. Use non-SCF rows,
  because every [SAFS] number is non-SCF.

[SAFS] Table 3 layout, ours next to theirs (agentic2026 split, non-SCF unless stated; [SAFS] values from §10.3):

| functional | train | val | test | WRMSD_tot | vs ωB97M-V |
|---|---|---|---|---|---|
| ωB97M-V (ours, SCF reference) | 3.319 | 4.086 | 3.632 | 4.090 | — |
| ωB97M-V [SAFS] | 3.32 | 4.02 | 3.64 | 4.07 | — |
| GAS22 (ours) | 3.270 | 3.778 | 3.168 | 3.788 | −7.4 % |
| GAS22 [SAFS] | 3.27 | 3.74 | 3.19 | 3.77 | −7.4 % |
| **it159 (ours)** | 3.252 | 3.736 | **3.307** | **3.776** | **−7.7 %** |
| SAFS26-a / SAFS26-b [SAFS] | 2.76 / 2.75 | 3.65 / 3.78 | 3.41 / 3.44 | 3.70 / 3.83 | −9.1 % / −5.9 % |
| AFS26-a / AFS26-b [SAFS] | 2.81 / 2.65 | 3.74 / 3.65 | 3.78 / 3.96 | 3.90 / 3.90 | −4.3 % / −4.2 % |

- Our GAS22 reproduces [SAFS] to within 0.04 in every column. The split is only approximate (whole C20C24 moved, not
  "two molecules"; our ωB97M-V val is 4.086 vs 4.02), so "vs ωB97M-V" is the safest column to compare across papers.
- it159 was selected on SyFES validation, where C20C24 was in test. Part of its agentic2026 validation score is
  therefore on data its selection never saw.
- Reading the table: it159 has a slightly lower WRMSD_tot than GAS22, beats SAFS26-b and both AFS26, and trails SAFS26-a.
  On test it beats every SAFS26/AFS26 functional (3.31 vs 3.41–3.96) but not GAS22 (3.17).

[SAFS] Table 2 layout: WRMSD_tot by category, non-SCF (it159 / GAS22 ours / ωB97M-V ours; [SAFS] GAS22 / ωB97M-V):

| category | n | it159 | GAS22 | ωB97M-V | [SAFS] GAS22 | [SAFS] ωB97M-V |
|---|---|---|---|---|---|---|
| BH | 136 | **7.31** | 7.77 | 8.28 | 7.75 | 8.28 |
| ID | 88 | 11.51 | **10.83** | 13.20 | 10.93 | 13.25 |
| IE | 679 | 3.02 | **2.62** | 2.97 | 2.65 | 2.98 |
| NCD | 69 | 2.43 | 2.38 | **2.36** | 2.37 | 2.36 |
| NCEC | 223 | **5.35** | 5.79 | 6.34 | 5.81 | 6.35 |
| NCED | 1433 | 2.16 | **2.10** | 2.25 | 2.10 | 2.25 |
| TCD | 258 | 2.66 | 2.84 | **2.56** | 2.36 | 1.99 |
| TCE | 660 | **3.91** | 4.16 | 3.97 | 4.16 | 3.97 |
| RG10 (not in WRMSD_tot; validation only) | 569 | 2.45 | **2.30** | 3.07 | 2.17 | 2.72 |

Source: `runs/scf_wrmsd_tot_agentic2026/wrmsd_tot.json` (`agentic2026` → `it159 nonSCF` / `GAS22 nonSCF` /
`wB97M-V SCF`; RG10 from the `validation` block). Bold = best of our three. Our values match [SAFS] to ≤ 0.1 except
TCD (see above) and RG10 (ours 3.07 / 2.30 vs their 2.72 / 2.17). Neither mismatch has been investigated.

## 13l. Where the per-iteration reasoning and token counts are stored (2026-10-26)

**Reasoning for each iteration**
* `runs/iterations.sqlite`, table `iterations`:
  * column `plan`: written by the agent before evaluation (the change vs the parent, and why);
  * column `summary`: written after evaluation (outcome, facts only).
* `runs/iterations.sqlite`, table `memory`: reasoning that spans many iterations (dead_end, insight, note).
* Exports:
  * `runs/nonscf_campaign_2026-10-15/iterations_1-200_full.jsonl` and
    `runs/nonscf_continuation_2026-10-19/iterations_201-400_full.jsonl` (full rows incl. plan and summary);
  * `global_memory.json` in both folders.
* `candidates/README.md`: plan and summary condensed to one row per iteration.
* MCP tools: `get_iteration(id)`, `get_lineage(id)`.
* Raw agent transcripts: everything not written into the plan. They live in
  `~/.claude/projects/-home-ash-Documents-dft-functional-sr/`:
  * campaign 1: `d3063582-21d8-4b7d-b69b-b36c76a74788.jsonl`;
  * campaign 2 lead: `e1a4e092-8583-4b73-91e4-56376cd7f89b.jsonl`;
  * campaign 2 island workers: `e1a4e092-8583-4b73-91e4-56376cd7f89b/subagents/agent-*.jsonl` (+ `.meta.json`).

**Token counts for each iteration**
* `runs/iterations.sqlite`, table `iterations`, columns:
  * tokens_input, tokens_output, tokens_cache_read, tokens_cache_creation, tokens_total;
  * cum_tokens_total, session_id.
  
  Method: §8.2.
* `runs/token_usage_per_iteration.csv`: the same columns, plus the agent that submitted each iteration.
* `iterations_1-200.csv` / `iterations_201-400.csv`: tokens_total and cum_tokens_total only.
* MCP tools: `get_token_usage`, `search_stats`.
* The raw transcripts above, the source of all counts.

## 13m. SyFES non-LLM baseline: regularized evolution from GAS22 (2026-11-05)

**Purpose.** Run a non-LLM comparison on the same data: SyFES (Ma et al. 2022) restarted from GAS22, with 400 trained forms. That is the same budget as the LLM campaigns (ids 1–400).

**Code.**
* `dftsr/syfes.py`: port of the SyFES symbolic machinery, without JAX. It covers:
  * the instructions and the SI instruction pool (arithmetic 0.06 ×5, power 0.05 ×6, UTransform 0.1, PBEX/RPBEX/B88X/PBEC 0.075 each);
  * the 4 mutation rules (0.25 each), component probabilities 1/3, at most 2 bound parameters per enhancement factor;
  * fingerprints for functional-equivalence checks;
  * GAS22 encoded as instructions. It matches the colab closed forms to 1e-14 and uses the SyFES ωB97M-V "short" parameter/variable counts (x 3/3, css 5/5, cos 6/5).
* `dftsr/syfes_gpu.py`: batched GPU CMA-ES.
  * Parameter-independent per-point quantities are precomputed once as "rows" in `<DATA_ROOT>/feature_cache/syfes_rows/<tag>/{train,validation}`, 3.2 + 7.2 GB, built in about 35 s:
    * closed-shell rows (x2, w, p_x, p_ss, p_os);
    * open-shell same-spin rows;
    * open-shell opposite-spin rows.
  * Rows are padded per molecule to 128-row blocks. A form becomes generated Python that `torch.compile` fuses into one kernel per row type, evaluated for all CMA-ES samples of all 5 restarts at once (P = 5·λ).
  * `as_functional(form)` wraps a form as a `Functional` for the float64 evaluator and the constraint suite.
* `scripts/syfes_run.py`: population server plus one GPU worker per GPU (asynchronous). Writes `history.jsonl` (resumable by replay) and `progress.json`.
* `scripts/syfes_recheck.py`: float64 re-check of the seed plus the top 10: exact train/val with `dftsr/evaluator.py`, stored parameters, and the 4 constraint tests. Writes `recheck.json` and `REPORT.md`.
* `scripts/syfes_job.sh`: runs the search, then the re-check.
* `dftsr/syfes_jobs.py`: job registry `runs/syfes/jobs.json`, plus summaries and lineage.
* MCP tools:
  * `syfes_submit` and `syfes_resume` launch a detached `systemd-run --user` unit and return immediately. They can idle up to `wait_seconds` ≤ 540 s.
  * `syfes_status(job_id, wait_seconds)` idles (async, polling every 20 s) until the job finishes or the wait passes, so it never reaches the 600 s MCP timeout.
  * Also `syfes_report`, `syfes_lineage`, `syfes_stop`.

**Protocol.**
* Taken from the SyFES paper and `example/config.py`:
  * population 100, tournament 25, mutation probability 0.9 (otherwise the tournament winner is re-added);
  * CMA-ES per new form on train: 5 restarts from N(0,1), σ0 = 1, bounds [−10, 10], tolfun 1e-5, max 1e5 evaluations per restart;
  * early stop if all values are ≥ 1e5, or if the best value is still > 50 after 5000 evaluations;
  * fitness = validation WRMSD of the best-train restart;
  * a fingerprint hit re-enters the population with its cached fitness and is not trained;
  * non-finite children are dropped.
* Stop after 400 trained forms.
* Seed = GAS22 with published parameters: train 3.2702, val 3.4625.

**Deviations, documented.**
* Only the parameters actually used are optimized. SyFES also passed the unused shared slots to CMA-ES.
* NaN or inf losses become 1e10 (worst rank). This is what pycma ≤ 3.0, used by SyFES, effectively did. pycma 4.5 would resample them forever.
* A child whose fingerprint is already being trained on another worker is skipped.
* Asynchronous with 3 workers instead of 50.
* Fingerprints use our PW_MOD LDA prefactors.
* Precision is adaptive:
  * float32 accumulation (noise ~4e-5 kcal/mol, 1.1 ms per parameter set) while a restart's last generation spans more than 0.01 kcal/mol;
  * then float64 accumulation (noise ~6e-6, 3.2 ms per set).
  * The final train value is recomputed with float64 accumulation.
  * Measured: GAS22 train 3.270186 against the float64 reference 3.270184.

**Cost (measured).**
* GAS22 form from random starts:
  * after 20,000 evaluations per restart it reaches train ≈ 3.205, below the published 3.270 because the gammas are refit too, and it is still moving;
  * so viable forms tend to run toward the 1e5 cap, about 10–25 GPU-minutes each;
  * broken mutants stop after about 5,000 evaluations per restart (≤ 1 min).
* Job 1: `runs/syfes/job0001_gas22_400`, systemd unit `syfes-job0001`, started 2026-11-05 16:22 on GPUs 0, 1, 2.

**Result of job 1 (DONE 2026-11-06 12:52; report `runs/syfes/job0001_gas22_400/REPORT.md`, `recheck.json`).**
* Run size and cost:
  * 20.5 h wall time on 3 GPUs (≈185 s wall per trained form), 61.1 GPU-h, mean 550 GPU-s and 119k CMA-ES evaluations per form;
  * 400 trained forms, 368 finite and 32 non-finite;
  * 1079 children skipped by fingerprint (functionally equivalent to a known form), plus 4 skipped because the same form was already training. Equivalent hits ran at about 3–6 per trained form late in the run.
* **No form beat the seed.** GAS22 (published parameters) stayed the best on validation: float64 train/val 3.2702/3.4623. The best-validation trajectory had its last improvement at trained form 95; the next 305 forms improved nothing.
* Float64 re-check of the top 10 (stored parameters, SyFES split):
  * form 174: train/val 3.4441/3.5057, **0/4 violations**. It passes the UEG exchange test (0.0032), which GAS22 fails (0.0121). Lineage: GAS22 → 23 → 88 (both equivalent to GAS22) → 174. So it is effectively one f_css mutation of GAS22: an extra `+ u²` term with coefficient fixed at 1 (u = γx²/(1+γx²)), plus a full 14-parameter refit.
  * forms 55, 699, 30, 1244, 11, 57, 784, 1266 and 542: val 3.554–3.595, all failing ueg_exchange.
    * Most add a cube or 6th power in f_x, or an extra constant.
    * They fit train better than GAS22 (3.188–3.207) and validation worse. This is the same overfitting pattern as the LLM campaigns' unconstrained branches.
* **Comparison with the LLM campaigns** (same SyFES split, same validation WRMSD, non-SCF):

  | search | best new form | val | constraints |
  |---|---|---|---|
  | SyFES, 400 forms | 174 | 3.506 | 4/4 |
  | LLM campaign 1 (ids 1–200) | it159 | 3.332 | 4/4 |
  | LLM campaign 2 (ids 201–400) | it396 | 3.138 (robust 3.185) | 4/4 |

  GAS22 itself scores 3.462. Over the same 400-form budget, SyFES restarted from GAS22 found nothing better than its seed. The LLM campaigns found forms 0.13–0.32 kcal/mol better on validation. Caveat: the LLM campaigns also used validation for selection over many iterations, and on test it159 loses to GAS22 (SCF 3.805 vs 3.589, `runs/scf_it159/comparison_table.md`; non-SCF test in §13f).
* Tooling fixes made during the run (they do not affect the results):
  * the ETA in `progress.json` divided by the GPU count twice;
  * the live `n_readds` counter was never incremented (the history file was correct);
  * `syfes_jobs.lineage` did not follow forms that entered via a fingerprint-equivalent mutation.

  All three are fixed in `scripts/syfes_run.py` and `dftsr/syfes_jobs.py`.

**Job 1 timeline and compute (full record).**
* Timeline, from session timestamps (local EST):
  * 2026-11-05 15:43: the user asks for the estimate;
  * 15:48: go-ahead;
  * 15:48–16:22: build and validation (≈35 min wall): the symbolic port, the GPU CMA-ES path with row precompute, the precision study, the GAS22 reference checks, the end-to-end resume/recheck test, and the MCP tools;
  * 16:22:23: `syfes-job0001` started;
  * 2026-11-06 12:52:10: 400th form trained;
  * 12:52:56: float64 re-check + `REPORT.md` written (≈46 s for 11 forms, train+val, plus constraints);
  * 12:52:57: unit exited.
  * The search wall time was 20 h 29.8 min (73,787 s).
* Hardware:
  * 3 × NVIDIA RTX 6000 Ada (48 GB), at 95–100 % utilization throughout, with about 28–33 GB used per GPU by the SyFES worker. The codex MCP server shared the GPUs, holding about 14–16 GB each.
  * The unit used 2 d 13 h 14 min of CPU time (population server + 3 workers).
* GPU cost:
  * 219,997 GPU-s = 61.1 GPU-h in training;
  * per trained form: mean 550 s, median 545 s, max 1,615 s;
  * 51 forms ended in < 60 s (broken forms stopped early);
  * wall time per form averaged 184 s (range 144–227 s per hour-long window).
* CMA-ES evaluations: 47.7 M in total (mean 119k per form, 5 restarts × up to 1e5 each).
* Storage:
  * parameter-independent rows at `<DATA_ROOT>/feature_cache/syfes_rows/` = 23 GB on disk (all row types, train + validation), built once in about 35 s;
  * history 3.9 MB.
* Search statistics:
  * 400 trained forms, 368 finite and 32 non-finite;
  * mutated component: f_cos 156, f_css 130, f_x 114;
  * mutation type: insert 148, change_argument 133, replace 94, remove 25;
  * 1,079 fingerprint-equivalent children (not trained, cached fitness reused), 167 re-adds (the 10 % no-mutation branch), 4 in-flight duplicates;
  * finite forms by validation: 30 below 3.6, 81 below 4.0, 42 above 10.
* Hourly progress (trained forms by local time): 16:55 8, 18:03 31, 19:03 50, 20:02 68, 21:01 86, 22:00 109, 23:00 130, 23:59 153, 00:58 171, 01:57 191, 02:56 207, 03:55 229, 04:54 245, 05:53 262, 06:52 279, 07:51 298, 08:50 316, 09:49 333, 10:48 357, 11:47 381, 12:46 399, 12:52 400.
* Initial estimate vs actual:
  * the estimate was ≈2–3 days including ≈1 day of build;
  * actual: ≈35 min build + 20.5 h run;
  * the GPU row-batching (all 5 restarts × λ in one kernel call, f32→f64 adaptive) is what made it fit in under a day.

**How comparable is this to the 2022 SyFES paper (Ma et al., Sci. Adv. 8, eabq0279 + SI).**
* Same as the paper:
  * regularized evolution (population 100, tournament 25, mutation probability 0.9, aging);
  * the SI instruction set and the 4 mutation rules;
  * fingerprint equivalence checking with cached fitness;
  * CMA-ES on train with 5 restarts from N(0,1), fitness = validation WRMSD of the best-train restart;
  * MGCDB84 with the paper's split (870/2960/1150 data points; train/primary test/secondary test of Mardirossian & Head-Gordon), the same WRMSD weights, non-SCF energies on fixed ωB97M-V densities.
  * Our GAS22 reproduction: val 3.462 non-SCF; test: published non-SCF 3.585 vs ours non-SCF 3.600 (§13f) and SCF 3.589.
* Different from the paper:
  1. **Starting point.**
     * The paper started from GAS22-a: the ωB97M-V form with all parameters incl. γ refit on train, train/val/test 2.97/3.82/4.47 (SI §5). It evolved down to GAS22 (val 3.46).
     * Job 1 started from GAS22 itself, the paper's final, validation-selected output.
     * So job 1 tests "can SyFES improve on its own published result", not "can it rediscover GAS22". The seed's val 3.462 is the minimum over the paper's entire search, so it is an optimistic bar.
  2. **Scale.**
     * The paper ran 12 independent evolutions with 50 GPU workers each and explored 29,628 forms in total (≈2,470 per evolution; whether that count includes fingerprint-equivalent children is not stated).
     * Job 1 is 1 evolution with 3 workers: 400 trained + 1,079 equivalent ≈ 1,480 proposed children. That is ≈5 % of the paper's total, about half of one of its evolutions.
  3. **Parameter-fit asymmetry.**
     * The seed keeps its published parameters. Every child gets only a fresh train-only CMA-ES fit.
     * Our CMA-ES on the GAS22 form reaches train ≈ 3.205, lower than the published 3.270. So the published parameters sit at a different local minimum that happens to validate well.
     * The near-miss children (train ≈ 3.19, val 3.57–3.59) may largely be GAS22 refits plus an almost-inert term.
     * The validation of a plain CMA-ES refit of GAS22 (≈15 GPU-min) was not measured.
  4. **Minor implementation deviations**, listed under "Deviations" above: used-only parameters, NaN → 1e10, 3 asynchronous workers, PW_MOD fingerprint prefactors, adaptive f32/f64 accumulation.
* Conclusion:
  * Job 1 is a fair equal-budget comparison against the LLM campaigns: 400 trained forms, same split, metric, non-SCF frame and functional family.
  * It is **not** a replication of the paper's B97M-V → GAS22 evolution.
  * Not done (optional follow-ups):
    * GAS22-form refit check (≈15 GPU-min);
    * a run from GAS22-a with ≥ 400 forms (≈20 h on 3 GPUs) to check that the port reproduces the paper's improvement trend.

## 13n. Paper revision: answering "there is no why" for the O159 window (2026-11-07, no new runs)

**Critique received** (paper at `/home/ash/Documents/dft_sr_paper/DFT_functional_agentic_search/achemso-demo.tex`):
- The paper gives no reason for the density window: no mechanism, an exponent that looks arbitrary (3 vs 4 differs by
  0.0006 kcal/mol), and an effect of about 0.15 kcal/mol in one refitted ablation.
- So it offers JACS readers neither a new physical idea nor a better tool.
- Suggested route A: link the window to a specific error pattern with a refitted control, then test whether that
  explanation predicts behaviour on independent examples.

**Constraint:** no new SCF runs, searches or fits. Everything below comes from records that already exist:
- `runs/iterations.sqlite` (fitted parameters, per-type and per-subset metrics);
- `candidates/README.md`;
- the GSCDB job reports.

The paper currently uses only two controls: it178 (no window) and the cubic vs quartic denominator.

**Scripts and outputs:** `runs/window_design_rule/`
- `window_error_decomposition.py` → `.txt`. Usage: `python … 159 145`; it compares each listed iteration with it178.
- `hatom_self_correlation.py` → `.txt`.
- `window_descriptor_analysis.py` → `window_descriptor_analysis.json`, `window_descriptor_bins.npz` (item 7.2, below;
  GPU, ~30 s).
- `make_window_figure.py` → `window_mechanism.png` / `.pdf`. It needs matplotlib, which is not in `.venv`, so run it
  with `/usr/bin/python3`.
- All are post-processing only: no fit, no SCF.

### What has to be added to the paper

**1. A new subsection, "What the window encodes: a design rule from refitted controls"**, with this table. Every row
was refit. Changes are in validation WRMSD (kcal/mol) relative to it145 (3.333), unless another parent is given. The
parent must be stated in the paper, because some controls come from earlier lineages.

| change | iteration (parent) | Δ val | conclusion |
|---|---|---|---|
| window removed | 178 (145) | +0.148 | carries ~75 % of 145's gain over ωB97M-V |
| window on the opposite-spin channel only | 185 (178) | 3.682 (+0.349; worse than no window) | same-spin specific |
| one shared window on ss and os | 186 (178) | 3.569 (+0.236) | not a general damping of correlation |
| total-density r_s instead of r_s,σ | 114 (92) | +0.074 | spin-resolved length scale |
| window on the ss uniform-gas term only | 136 (125) | +0.331 | must scale the whole ss factor |
| broad r/(1+r²) / exp cutoff / symmetric h₄ | 184 / 173 / 150 (145) | +0.087 / +0.058 / +0.034 | a sharp low-density cutoff matters |
| peak r₀ = 0.75 / r₀ = 1.3 / trainable r₀ | 137 / 138 (125); 96 (92) | +0.015 (flat) / +0.083 / r₀ stays at 1 | located at r_s,σ ≈ 0.75–1 bohr |
| gate off in one-orbital regions (1 − τ_W/τ) | 166 (152) | +0.156 | suppression needed where τ_W/τ is large |
| gradient radius / kinetic radius r_τ | 189 / 163, 164 (152, 145) | +0.126 / neutral | extra information is kinetic, not gradient |
| exact uniform-gas ss limit | 169 (125) / 170 (145) | +0.29 / +0.58 | ss correlation must sit well below the uniform-gas value |
| constant amplitude only | 152 (145); 191 | +0.011; β₀ −0.37 to −0.39 from every start | one stable physical parameter |
| transfer to ωB97M-V / GAS22 ss polynomials | 27, 60, 98, 143 | transfers | not tied to one polynomial |
| transfer to pure meta-GGA (B97M-V) | 161 vs 160 | 6.189 → 6.154, weak | specific to the hybrid frame |

**Rule to state:**
- In this range-separated hybrid, scale the same-spin enhancement factor toward zero by about 15–30 % in a valence
  band of spin density (r_s,σ ≈ 0.75–1 bohr).
- Precise wording matters: with the GAS22 ss polynomial, g_ss changes sign with w. So the window *lowers* the energy
  where τ < τ_UEG (w > 0, where g_ss·e_c^LDA > 0) and *raises* it where τ > τ_UEG. It shrinks the magnitude of the ss
  factor; it does not simply "remove correlation". Summed over the validation molecules, the window energy is
  −59,000 kcal/mol for w > 1/3 and +88,000 for w < −1/3 (item 7.2).
- Apply the cut to the whole ss enhancement factor, with a sharp low-density cutoff.
- Do not apply it to opposite-spin correlation.
- Magnitude: h = r/(1+r⁴) peaks at 0.57 (r = 0.76). The factor is ≈ 0.85 at w = 0 (β₀ = −0.255) and ≈ 0.70 at
  |w| = 1 (β₀ + β₂ = −0.52).

**Noise caveat to state:** fits depend on the starting point. Form 145 spans 3.33–3.43 across starting parameters
(it187, it188), so treat only |Δ| ≳ 0.05 as meaningful.

**2. Reframe the two weak sentences in §3.1:**
- **Exponent:** the controls show that location and cutoff cost 0.03–0.09 when changed, while 3 vs 4 costs 0.0006.
  The rule fixes position and cutoff, not the exponent. Present this as a robustness result.
- **Effect size:** 0.148 kcal/mol is about twice GAS22's own validation gain over ωB97M-V (3.533 → 3.462 = 0.071).
  It is also ≈ 75 % of O145's gain.

**3. An error-pattern figure** (bar chart) from the it178 → it159 decomposition of the change in WRMSD². Category
shares are exact (from the stored per-type WRMSD). Subset shares are approximate (n · mean weight · RMSD², rescaled
within each category).

*Validation, J 3.481 → 3.332:*
- By category: BH 35 %, RG10 22 %, TCE 19 %, ID 14 %, NCEC 12 %; NCED −8 %.
- Helped: Shields38 0.49 → 0.35; HTBH38 1.71 → 1.49; HAT707nonMR 2.67 → 2.59; WATER27 0.49 → 0.42;
  ISOMERIZATION20 2.32 → 2.18; CR20 0.81 → 0.57.
- Hurt: SW49Bind6 0.55 → 0.70 and SW49Bind345 0.25 → 0.32 (sulfate–water anion clusters); BzDC215; S66x8;
  EA13 1.78 → 1.92.

*Training, J 3.386 → 3.252:*
- AE18 gives 62 % of the gain (RMSD 7.60 → 5.56).
- Also helped: Styrene45, BHPERI26, DBH24.
- Hurt: CRBH20 1.02 → 1.37, G21EA.

*Reading:*
- The window helps where compact valence correlation changes: atoms, H-transfer transition states, water networks.
- It costs for anions (EAs, sulfate–water binding) and dispersion-bound dimers.
- **Correction from item 7.2:** the anion losses are *not* a diffuse-tail effect. The harmful contribution for
  SW49Bind sits in the same valence band (r_s,σ 0.8–1.5, w > 1/3) as the gains; the r_s,σ > 3 tail contributes
  slightly negatively.

*Flag in the caption:* the 22 % RG10 share comes from weighting (RMSD 0.183 → 0.182) and is not physics.

**4. A rejected explanation (keep it in; it strengthens the paper).** "The window corrects same-spin
self-correlation" does not hold. The exact H atom (fully polarised, exact density) has E_c = 0. Computed values:

| functional | E_c(H), kcal/mol |
|---|---|
| ωB97M-V | +4.11 |
| GAS22 | +4.36 |
| it178 (refitted, no window) | +2.50 |
| it145 | +4.14 |
| it159 | +3.52 |
| it159 with the window off (same other parameters) | +4.55 |

The window reduces it159's own error, but the refitted control without the window has less. So the gain is not a
self-correlation fix.

**5. Hypothesis, worded as a hypothesis, not a mechanism.**
- Candidate wording: same-spin correlation taken from the polarised uniform gas (Stoll partition of PW92)
  overestimates parallel-spin correlation in compact valence shells. The B97 w/u polynomial cannot localise that
  correction in density; the window can.
- Sharpened by item 7.2: the accuracy that depends on the window comes from intermediate-density regions where
  τ < τ_UEG/2. These are single-orbital-like regions such as the density near protons and in covalent bonds. For the
  exact H atom, r < 0.5 bohr maps to r_s,σ 0.91–1.27 and w 0.62–0.37, inside the zone.
- This fits the held-out test gains (water clusters, proton-transfer barriers).
- Do **not** phrase it as self-correlation (item 4 rejects that). Phrase it as the descriptor zone in which the B97
  same-spin factor needs a density-localised rescaling.
- List what the controls exclude:
  - channel-agnostic damping (185, 186);
  - a gradient effect (189);
  - a uniform-gas renormalisation (136, 169, 170);
  - self-correlation (item 4).

**6. A direction check on independent data** (GSCDB137, the 20 sets without MGCDB84 antecedents; job 0001
`result.json`):
- AE11 heavy-atom energies improve: MAE 369 → 296 kcal/mol.
- DBH22 1.17 → 1.15 and ORBH35 2.58 → 2.54 improve slightly.
- O24 improves: 1.03 → 0.90.
- P34EA EAs get worse (+0.21), as predicted for anions.
- Transition metals get worse (3d4dIPSS +0.54, TMD10 +0.55, DAPD +0.36). The pattern did not predict this; say so.
- **Confound to state:** this is it159 vs ωB97M-V, not vs the no-window control, so the window and the refit of the
  other coefficients cannot be separated.
- **Resolved by item 7.1:** against the refitted no-window control, the AE11 gain is the window's (415.8 → 295.8).
  DBH22 also improves with the window. The TM sets are slightly better with the window than without, so the TM
  losses vs ωB97M-V come from the refit, not the window. P34EA does get worse because of the window.

**7. Limitations: name the two missing pieces.**
1. ~~The it178 no-window control on GSCDB, non-SCF.~~ **Done 2026-11-07** via the semilocal-difference shortcut
   (3 × ~9 min ≈ 0.5 GPU-h, not 7); results under "Item 7.1 results" below.
2. ~~A per-atom and descriptor-resolved breakdown of the window energy.~~ **Done 2026-11-07 (user go-ahead);
   results below.**

**Expected outcome:**
- Before item 7.2: this moves the paper from "no why" to a controlled design rule with a tested, rejected
  alternative and a direction check on independent data.
- With item 7.2: it also has a descriptor-resolved localisation (5 % of electrons carry 82–91 % of the
  window-dependent accuracy) and a held-out check against the refitted control on the MGCDB84 test split.
- With item 7.1 (done 2026-11-07, results below): a refitted-control test on external data with predictions stated
  beforehand. The atomic-energy part transfers (AE11 −29 %); beyond it, no demonstrated benefit on the new sets.
- It is still not a mechanism.

### Item 7.2 results: where the window acts and what it fixes (2026-11-07)

**Setup:** `runs/window_design_rule/window_descriptor_analysis.py`.
- Archived parameters on the cached ωB97M-V/def2-QZVPPD densities: float32 cache, float64 arithmetic, legacy
  attenuation kernel (§13o shows the kernel choice is irrelevant). No fit, no SCF; ~30 s on one GPU for all 5,930
  molecules.
- Window energy per molecule: W = E_c(it159) − E_c(it159, β₀ = β₂ = 0), i.e. Σ w·e_c,ss^PW92·P_s·(β₀ + β₂w²)·h(r_s,σ)
  over points and spin channels.
- It is binned per molecule on a grid of 31 log-spaced r_s,σ bins (0.05–20 bohr, plus overflow) × 20 w_σ bins.
- Closure Σ_bins W = E_on − E_off holds to ≤ 2e-11 Eh.
- Three models per reaction:
  - "on" = it159;
  - "off" = it159 with the window removed and the other parameters kept;
  - "ctrl" = it178, the refitted no-window control.
- Bin attribution of J_on² − J_off² is exact: Σ_b (1/N) Σ_i w_i (2e_off,i + W_i) W_ib.
- Per category/subset, the exact share of J_on² − J_ctrl² is also given.

**WRMSD (kcal/mol), on / off / ctrl:**

| split | O159 (on) | window removed, same params (off) | refitted control it178 (ctrl) |
|---|---|---|---|
| train | 3.252 | 5.111 | 3.386 |
| val | 3.332 | 5.595 | 3.481 |
| test | 3.818 | 7.188 | **3.865** |

- **New number for the paper:** the refitted control's *test* WRMSD is 3.865. So the window's held-out gain is 0.047
  (val gain 0.149), ≈ 1.2 % of test WRMSD.
- Removing the window at fixed parameters is catastrophic (val 5.60). The fitted O159 depends on it heavily, and a
  refit without it recovers most of the loss.

**(1) AE18 atoms, error in total energy (kcal/mol), ctrl → O159; W = window energy:**

| | H | He | Li | Be | B | C | N | O | F | Ne | Na | Mg | Al | Si | P | S | Cl | Ar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | 2.25 | 0.94 | 0.62 | −0.30 | −1.74 | −5.38 | −7.02 | −12.17 | −13.54 | −9.90 | −7.44 | −10.66 | −6.85 | −1.74 | 5.48 | 6.13 | 8.14 | 13.00 |
| O159 | 2.98 | 1.65 | 2.51 | 4.22 | 4.81 | 1.54 | −0.56 | −6.81 | −9.26 | −7.11 | −6.25 | −9.79 | −6.26 | −1.74 | 4.48 | 4.67 | 5.25 | 8.15 |
| W | −0.83 | −2.62 | −2.07 | −0.32 | 0.66 | 1.72 | 3.20 | 1.63 | 2.19 | 3.54 | 3.69 | 4.54 | 5.62 | 6.81 | 8.22 | 8.89 | 9.73 | 10.79 |

- The control has a systematic **shell-structure error**: 2p atoms overbound (to −13.5 at F), 3p atoms underbound
  (to +13.0 at Ar).
- With the window, both ends shrink: |error| of C–F falls by 3.8–6.5 kcal/mol (N −7.0 → −0.6); Cl 8.1 → 5.3,
  Ar 13.0 → 8.2.
- The cost is the light atoms: |error| of H–B grows by 0.7–3.9 (Be −0.3 → 4.2).
- The window energy W grows with Z overall (−2.6 for He to +10.8 for Ar; not strictly monotonic, e.g. O < N).
- So the gain is *not* specific to high-spin atoms; the 13n guess "high-spin N and P" is wrong for P (5.48 → 4.48
  only).
- Per region, the window energy of 2p atoms sits mostly in r_s,σ 0.4–0.8 (e.g. N +4.5, O +4.9 kcal/mol), partly
  offset by core r_s,σ < 0.4. For 3p atoms the positive part moves to 0.8–1.5.

**(2) Descriptor localisation** (fixed-parameter attribution of J_on² − J_off²; `window_mechanism` panel c):

| split | share from r_s,σ 0.4–1.5 bohr | share from w > 1/3 (τ < τ_UEG/2) | share from both | electrons in both |
|---|---|---|---|---|
| val | 87 % | 82 % | 82 % | 4.9 % |
| test | 88 % | 91 % | 91 % | 5.4 % |

- About **5 % of the electrons carry 82–91 % of the window-dependent accuracy**.
- Core (< 0.4) and tail (> 3) contribute ≈ 0 or slightly against.
- By category: HTBH38 98 % in 0.8–1.5; Shields38 and WATER27 mostly in 0.4–1.5 with w > 1/3.
- The SW49Bind losses also sit in 0.8–1.5 with w > 1/3 (see the item 3 correction).
- Caveat for the figure: neighbouring bins carry large values of opposite sign, so only the region totals are robust,
  not individual cells.

**(3) Held-out check against the refitted control** (MGCDB84 test split, never used for selection; exact shares of
J_on² − J_ctrl² = −0.361):
- By category: NCEC −0.487, IE −0.357, BH −0.239, TCE +0.010, TCD +0.028, NCED +0.260, ID +0.423.
- Helped: H2O20Bind4 1.26 → 0.69, YMPJ519, WCPT27 1.86 → 1.70, PX13 1.92 → 1.82, H2O20Bind10.
- Hurt: C20C24 5.39 → 5.72 (8 reactions), Ionic43 0.61 → 0.67, BSR36.
- **The validation pattern predicts the test pattern:**
  - water clusters and H/proton-transfer barriers improve;
  - ionic (anion-containing) binding gets worse, the test-set counterpart of SW49Bind;
  - TCD is neutral to slightly worse.
- This is the route-A "predicts behaviour on independent examples" result, from existing data. Caveat: the
  hypothesis was formed after the test numbers in §13f were known, so present it as a consistency check, not a
  pre-registered prediction.

**For the paper (items 1, 3, 5, 7):**
- Add `runs/window_design_rule/window_mechanism.pdf` as the mechanism figure, or panels (a) and (d) in the main
  text with (b) and (c) in the SI.
- Add the test-split control number (3.865) next to the val ablation (3.481).
- Use the corrected rule wording (shrinks the magnitude of the ss factor) and the corrected anion statement.
- Not yet written into the LaTeX.

### Item 7.1: window control on GSCDB137, predictions written before computing (recorded 2026-11-07 22:32 EST)

**Comparison:** it178 (refitted no-window control) vs it159, both non-SCF on the stored ωB97M-V orbitals, 36-set
subset. The quantity is ΔNER_set = NER(it159) − NER(it178); negative means the window helps.

**Scoring rule:**
- "neutral" = |Δmetric| / metric(it178) < 1 %.
- "improve" / "worsen" otherwise, by sign.
- A directional prediction counts as a hit if the sign matches and the set is not neutral.
- A "neutral" prediction counts as a hit if the set is neutral.
- No-prediction sets are reported but not scored.
- Primary aggregate: mean NER over the 20 new sets, with a paired Wilcoxon test and a bootstrap CI over sets.

**Basis of the predictions** (from §13n, item 7.2):
- The window helps atomic total energies (shell-structure trend), H- and proton-transfer barriers, water and H-bond
  networks, and BH76RC / HAT / isomerisation-type thermochemistry.
- It hurts anion-containing binding and electron affinities, and slightly hurts dispersion-bound dimers.
- It says nothing about transition metals, field response, frequencies, or one-electron SIE systems.

**20 new sets** (no MGCDB84 antecedent; this group is the real test):

| set | content | prediction | confidence |
|---|---|---|---|
| AE11 | heavy-atom total energies (Ca … Rn) | improve | low (extrapolates the AE18 trend) |
| IP23 | ionisation potentials | improve | low (IP13 improved, G21IP neutral) |
| P34EA | EAs of heavy p-block atoms | worsen | medium (EA13, G21EA worsened) |
| HCP32 | halogen/chalcogen/pnictogen bonds | improve | low (XB51, XB18 improved) |
| O24 | radical–molecule NC complexes | worsen | low (NC dimers slightly worse) |
| O24x4 | same, off-equilibrium | worsen | low |
| He3 | He trimers | neutral | medium (energies tiny) |
| ALKBDE10, DIPCS9, MB08-165, SIE4x4, HEAVY28 | — | no prediction | |
| 3d4dIPSS, DAPD, TMD10, ORBH35 (TM) | — | no prediction | |
| Dip146, OEEF, Pol130 (field), V30 (freq.) | — | no prediction | |

**16 updated MGCDB84 sets** (not independent; they check that the MGCDB84 control pattern reproduces):

| set | MGCDB84 role / control result | prediction |
|---|---|---|
| AE18 | train, 7.60 → 5.56 | improve |
| DBH22 | train (DBH24 1.58 → 1.46) | improve |
| WCPT6 | test (WCPT27 1.86 → 1.70) | improve |
| BH76RC | val, 1.41 → 1.36 | improve |
| HAT707nonMR | val, 2.67 → 2.59 | improve |
| HAT707MR | val, 4.95 → 4.88 | improve (weak) |
| ISOMERIZATION20 | val, 2.32 → 2.18 | improve |
| SN13 | val, 0.85 → 0.67 | improve |
| TAE_W4-17MR | val (TAE140MR 10.52 → 10.15) | improve |
| BDE99nonMR | train, 2.55 → 2.52 | improve (weak) |
| BDE99MR | val, 5.76 → 5.71 | neutral |
| G21EA | train, 2.48 → 2.66 | worsen |
| TA13 | train, 2.63 → 2.63 | neutral |
| RG10N | val, 0.183 → 0.182 | neutral |
| NC11 | val (NC15 0.054 → 0.054) | neutral |
| CT20 | val, 0.106 → 0.103 | neutral |

**Overall expectation, stated beforehand:**
- A small or null aggregate effect on the 20 new sets: the MGCDB84 test gain is only 0.047 kcal/mol.
- Most of the information is in the per-set directions.

### Item 7.1 results: window control on GSCDB137 (computed 2026-11-07, after the predictions above)

**Method:**
- E_nonSCF(it178) = E_nonSCF(it159, job 0001, full precision from molecules.jsonl) + [E_sl(it178) − E_sl(it159)] at
  the stored ωB97M-V density. The bracket is integrated on the production grids by a pair-difference NumInt.
- Script `runs/window_design_rule/gscdb_it178_delta.py`: 3 systemd shards, ~9 min each, 0 errors. AE11_Rn uses the
  CPU grid (`--cpu`); the GPU value was −1.4e9 Eh, the known Rn defect.
- Scoring: `gscdb_it178_summary.py` → `gscdb_it178_summary.json`.

**Checks:**
- The shortcut equals the production `candidate_nonscf` it178 energy to ≤ 3e-14 Eh on 5 molecules (closed-shell,
  open-shell, field, TM, H atom).
- The it159 recomputation reproduces archived `result.json` NERs exactly (max |ΔNER| = 0).
- Archived ('legacy') attenuation kernel, as in job 0001 (§13o: irrelevant).

**Mean NER, non-SCF** (Δ = it159 − it178; negative means the window helps; bootstrap 95 % CI over sets, 20k
resamples; Wilcoxon signed-rank):

| scope | it178 (no window) | it159 | ωB97M-V | Δ | 95 % CI | p | sets better / worse |
|---|---|---|---|---|---|---|---|
| all 36 | 1.189 | 1.141 | 1.144 | −0.048 | [−0.121, +0.011] | 0.041 | 26 / 10 |
| 16 updated MGCDB84 sets | 1.036 | 0.979 | 0.961 | −0.057 | [−0.109, −0.010] | 0.018 | 12 / 4 |
| **20 new sets** | 1.312 | 1.270 | 1.290 | −0.041 | [−0.171, +0.054] | 0.55 | 14 / 6 |
| 19 new sets without AE11 | 1.194 | 1.204 | 1.191 | +0.011 | [−0.046, +0.072] | 0.80 | 13 / 6 |
| 35 sets without AE11 | 1.122 | 1.101 | 1.086 | −0.020 | [−0.060, +0.021] | 0.067 | 25 / 10 |

**Per-set changes** (relative change of the dataset metric, it178 → it159):
- New sets, better with the window: AE11 −28.9 % (415.8 → 295.8), MB08-165 −15.9 %, OEEF −14.1 %, O24 −7.6 %,
  DAPD −6.0 %, HEAVY28 −5.0 %, He3 −3.9 %, ALKBDE10 −3.6 %, ORBH35, TMD10, 3d4dIPSS, SIE4x4, O24x4, IP23 (−1 to −3 %).
- New sets, worse: V30 +29.6 %, DIPCS9 +22.1 %, P34EA +19.1 %, HCP32 +16.1 %, Dip146 +4.6 %, Pol130 +3.2 %.
- Updated sets, better: AE18 −22 %, G21EA −20 %, SN13 −17 %, BH76RC, DBH22, HAT707nonMR (−6 to −7 %),
  ISOMERIZATION20, CT20, BDE99nonMR.
- Updated sets, worse: WCPT6 +24.5 %, BDE99MR +2.1 %.

**Prediction scoring** (rule as recorded):
- New sets: **3 / 7 hits**.
  - Hits: AE11 improve, IP23 improve (−1.1 %), P34EA worsen.
  - Misses: O24 and O24x4 (predicted worse, improved), He3 (predicted neutral, −3.9 %), HCP32 (predicted better,
    +16 %).
- Updated sets: **11 / 16 hits**.
  - Misses: G21EA (predicted worse, improved 20 %; MGCDB84 G21EA worsened, so the GSCDB revised references may
    differ), WCPT6 (predicted better from the WCPT27 test gain, worsened 24.5 %), CT20 and NC11 (predicted neutral,
    small gains), BDE99MR (predicted neutral, +2.1 %).

**What this says:**
1. **The atomic-energy part of the mechanism transfers.** The strongest pre-stated extrapolation, AE18's
   shell-structure fix carrying over to heavy atoms (AE11, Ca … Rn), holds with the largest effect of any set
   (−29 %). AE18 itself improves 22 %.
2. **Beyond atomic energies, there is no demonstrated external benefit.** On the 20 new sets the mean gain is
   −0.041 with a CI spanning zero (p 0.55). Without AE11 the window is neutral to slightly harmful (+0.011).
   Directional predictions on new sets beyond AE11/IP23/P34EA failed (3/7 overall, about chance).
3. **The window brings the refitted frame back to parity with ωB97M-V on the 36 sets.** it178 1.189 vs ωB97M-V 1.144
   vs it159 1.141. Without the window, the refitted frame is clearly worse than ωB97M-V externally.
4. **The anion prediction holds on P34EA** (+19 %), consistent with SW49Bind and Ionic43. It fails on G21EA (−20 %).
5. **Costs the paper should state:** vibrational frequencies (V30 +30 %), double ionisations (DIPCS9 +22 %),
   σ-hole bonds (HCP32 +16 %), field response (Dip146/Pol130 +3–5 %). No pre-stated prediction covered these.

**How to write it in the paper:**
- Report as a refitted-control test with predictions stated beforehand (record the timestamp).
- The window's external value is clear for atomic total energies, aggregate-positive but not robust over the 36
  sets (p 0.04 but CI crosses zero), and not demonstrated on the new sets once AE11 is excluded.
- This removes the confound in item 6: the AE11 improvement is attributable to the window, while the TM losses vs
  ωB97M-V are not (TM sets are slightly *better* with the window than in the control).
- Do not claim the design rule "predicts" external chemistry in general; it predicted the atomic-energy direction
  and the anion direction on one of two EA sets.

## 13o. Attenuation large-a branch: corrected-kernel re-evaluation (2026-11-07, "option B")

**The critique** (rated "critical"):
- The paper's SI and conclusion admitted an unresolved, unquantified exchange-attenuation discrepancy.
- So: the parameters were fitted to a functional that differs from Eq. S30, and the SCF comparison mixes
  implementations (libxc ωB97M-V reference vs torch candidates).
- All of these are one bug.

**Root cause** (`dftsr/xc_torch.py: f_rsh`):
- For a = ω/k_Fσ ≥ 5 the code used 1/(36a²) − 1/(960a⁴) + 1/(26880a⁶). That is the expansion for a' = ω/(2k_F), so it
  returns F(2a): 4× too small and discontinuous at a = 5 (0.0011094 vs 0.0044179).
- At ω = 0.3 this affects only 1e-10 < ρσ < 3.65e-6 au (tails).
- Upstream SyFES evaluates the closed form everywhere, so the published GAS22 parameters are unaffected. The bug is
  ours, from the torch port.

**Code change (archived behaviour unchanged):**
- `f_rsh` now reads the module global `RSH_LARGE_A_SERIES`, set from env `DFTSR_RSH_SERIES`.
- `legacy` (default) keeps the archived branch; `corrected` uses 1/(9a²) − 1/(60a⁴) + 1/(420a⁶) (matches mpmath to
  2e-10 at a = 5).
- The running MCP workers and every archived pipeline therefore behave exactly as before.

**Scripts and outputs:** `runs/attenuation_fix/`
- `reevaluate.py` → `reevaluation.json`, `per_molecule_shifts.json`. MGCDB84, all 5,930 molecules: ωB97M-V, GAS22,
  O159, O396 and C356 (C356 parameters read only from the Codex DB), float64, ~80 s on one GPU.
- `refit_o159.py` → `refit_o159.json`.
- `gscdb_delta.py` (3 systemd shards, ~25 min each) and `gscdb_delta_rn.py` (AE11_Rn on the CPU grid) →
  `gscdb_delta/*.jsonl`.
- `gscdb_summary.py` → `gscdb_summary.json`.
- No fitting except `refit_o159.py`. No SCF.

**Method:**
- **Non-SCF:** E_corr = E_arch + Δ_cand − Δ_ref, with Δ = E_sl(corrected) − E_sl(legacy) at the same density.
  - Written as a difference on purpose. The cached `E_sl_t64` (full-precision density) and a recomputation from the
    float32 cache differ by up to 1.3e-6 Eh, which would otherwise swamp the effect. The first attempt made that
    mistake: a spurious ωB97M-V val change of −3.6e-4.
- **SCF:** E_scf + Δ_cand at the ωB97M-V density. The first-order change of a variational energy equals the
  perturbation at the converged density; the density difference enters at second order.
- **GSCDB:**
  - Production grids from the stored ωB97M-V orbitals. A NumInt whose `eval_xc_eff` returns (e_corr − e_leg)/ρ, so
    `nr_rks`/`nr_uks` integrate Δ directly.
  - Full-precision energies come from `molecules.jsonl` via `gscdb_jobs.read_records`. `energies.csv` keeps 10
    decimals, and V30's second finite differences turn that into ~1e-4 NER.
  - Legacy recomputation reproduces every archived `result.json` column exactly (max |ΔNER| = 0).
  - AE11_Rn uses the CPU-grid value; the GPU value for GAS22 was wrong by 2× (the known Rn GPU-XC defect).

**Results** (all changes far below the reported precision):

| | train | val | test |
|---|---|---|---|
| O159 non-SCF ΔWRMSD | −2e-8 | −4e-7 | +7e-8 |
| O159 SCF ΔWRMSD | −4.7e-6 | −6.4e-5 | +5.6e-6 |
| GAS22 non-SCF / SCF ΔWRMSD | −3e-7 / +8.6e-6 | −9e-7 / +8.7e-6 | −3.8e-6 / +4.6e-5 |
| O396 / C356 non-SCF ΔWRMSD | −2e-8 / −6e-8 | −3e-7 / −3.7e-6 | +9e-8 / +6.4e-6 |
| ωB97M-V (libxc) | 0 | 0 | 0 |

*MGCDB84:*
- ~10 % of spin-channel grid points are affected.
- Max per-molecule candidate shift: 2.6e-4 kcal/mol (C24 ring; large, diffuse systems are largest).
- Max non-SCF reaction-energy change: 5.7e-5 kcal/mol (GAS22).
- H atom (analytic, `runs/window_design_rule/attenuation_check.py`): ΔE_x −3.07e-8 Eh (O159) vs −3.05e-8 Eh (torch
  ωB97M-V), so the non-SCF shift cancels.

*GSCDB137, 36 sets:*
- Max per-molecule shift: 4.6e-4 kcal/mol.
- Mean NER: it159 SCF 1.138725 → 1.138757; GAS22 SCF 1.190306 → 1.190356; it396 non-SCF 1.146814 → 1.146854.
- 20 new sets: it159 SCF 1.268628 → 1.268692.
- Worst set is V30 for every functional (finite-difference frequencies): GAS22 SCF 32.021 → 32.059, it159 SCF
  33.548 → 33.574; max |ΔNER| 1.25e-3.
- Paired tests unchanged: it159 − ωB97M-V −0.00492 → −0.00489 (p 0.58); GAS22 − it159 +0.05158 → +0.05160 (p 0.020).

*O159 refit (same init and settings as the archived fit):*
- The float32 compiled fit is **not bit-reproducible**: 4 cold fits per kernel land in two basins (val 3.3326–3.3382,
  β ≈ (−0.254, −0.267) or (−0.225, −0.360)), with within-kernel parameter spread 0.11 under both kernels.
- Warm start from the archived parameters: corrected and legacy optima differ by ≤ 4.4e-6 in every parameter; val
  3.332431 (legacy) vs 3.332403 (corrected), test 3.817962 vs 3.817956.
- 0 constraint violations under both kernels.
- The archived parameters can be used with the closed-form F.
- The potential kink at a = 5 (ρσ = 3.65e-6) is not the cause of the GSCDB convergence repairs (ρσ ≈ 1e-10).

**Paper edits** (`/home/ash/Documents/dft_sr_paper/DFT_functional_agentic_search`, uncommitted; LaTeX is not installed
here, so only brace, label and math-delimiter checks were run):
- SI subsection renamed "Numerical guards and the attenuation branch" (`sec:si-attenuation`): root cause, method,
  results, new Table `tab:si-attenuation`, refit paragraph.
- SI reproducibility sentence now points there.
- Main text:
  - Methods (frozen evaluation): one sentence on the torch vs libxc paths and the bound.
  - Limitations: "unquantified … revalidation required" replaced by the quantified statement.
  - Conclusions: the "unresolved attenuation" sentence removed.

**Not done:**
- The paper's reproducibility archive (`/home/ash/Documents/dft_sr_paper/reproducibility/`, `check_attenuation.py`
  says "no molecular-data impact was computed") does not yet contain these scripts or outputs.
- The other paper places that quote archived numbers are untouched, because none changes at the printed precision.

## 14. Command cheat-sheet

```bash
cd /home/ash/Documents/dft_functional_sr
.venv/bin/python scripts/density_progress.py                     # campaign progress
tail -f /media/ash/PortableSSD/dft_sr/post_density.log           # cache/baseline pipeline
tail -f /media/ash/PortableSSD/dft_sr/densities/wb97mv_def2qzvppd_L3/_logs/worker_gpu0.out
bash scripts/run_density_campaign.sh                             # (re)start density workers
pkill -f "python -u [s]cripts/compute_densities"                 # stop density workers
.venv/bin/python scripts/build_feature_cache.py --splits train,validation [--allow-partial]
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/establish_baseline.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dftsr.worker baselines/gas22.py [--nofit] [--test] [--restarts N]
.venv/bin/python scripts/scf_evaluate.py --candidate X.py --params p.json --subset probe --out DIR --gpus 1,2
claude mcp get dft-functional-search                             # registration status (user/global scope)
claude                                                           # MCP_TOOL_TIMEOUT=600000 is set globally in ~/.claude/settings.json
# GSCDB137 (normally via the MCP tools gscdb_*; the manual form, as a systemd user unit):
systemd-run --user --unit=gscdb-manual --collect --property=WorkingDirectory=$PWD \
  .venv/bin/python scripts/gscdb_scf.py --candidate candidates/00159_F145_r4.py --params p.json --label it159 \
  --subset probe|all|datasets [--datasets AE18,TMD10] [--max-nbas 3000] \
  --out /home/ash/Documents/dft_fucntional_sr_gold_std/runs/<dir> --gpus 0,1,2
.venv/bin/python runs/nonscf_test_388_376/run_test.py 396          # non-SCF test-split check of stored params
bash scripts/syfes_job.sh <outdir> 400 0,1,2 0 100 25 5            # SyFES baseline run + fp64 re-check (MCP: syfes_submit)
.venv/bin/python scripts/syfes_run.py --out <outdir> --n-forms 400   # rerun the same command to resume from history.jsonl
```
