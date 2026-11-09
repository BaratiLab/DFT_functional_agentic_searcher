# Agentic discovery of DFT exchange–correlation functionals: reader reproduction

This repository packages the computational workflow behind the paper in `DFT_functional_agentic_search`: MGCDB84 fixed-density screening, frozen-functional non-SCF and SCF evaluation, and external evaluation on the **36 reported GSCDB137 datasets**. It includes the Claude Code/Opus and Codex harnesses and the later non-LLM SyFES comparison. No paid agent service is needed to evaluate the frozen functionals.

There are three distinct reproduction tasks:

1. **Recompute reported metrics from saved energies** on a CPU, without rerunning electronic structure: `python tools/verify.py`.
2. **Recompute molecular energies** with the frozen source and fitted parameters, using the commands below. This requires external geometries, regenerated reference densities, and GPU resources.
3. **Run a new agent search** using the supplied prompts, MCP servers, and coordinator workflow in [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md). A new stochastic search is a new experiment; it will not necessarily rediscover the same candidates or reproduce the historical token ledger.

The large density/cache files and proprietary agent models are not distributed here. Original source provenance, exact copied-file hashes, and the assembly changes are in [provenance/source_manifest.json](provenance/source_manifest.json) and [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md). Archived absolute paths are historical metadata; the entry points under `tools/` use this checkout and `work/`.

## What was actually evaluated

| Frozen functional | MGCDB84 non-SCF | MGCDB84 SCF | GSCDB36 non-SCF | GSCDB36 SCF |
|---|---|---|---|---|
| ωB97M-V (`wB97M-V`) | Reference/implementation checks | Reference densities | Reference | Reference |
| Published `GAS22` | Yes | Yes | Yes | Yes |
| `O159` | Yes | Yes | Yes | Yes |
| `O396` | Yes | Not reported | Yes | Not reported |
| `C356` | Yes | Not reported | Not reported | Not reported |
| `O178`, refitted no-window control | Yes | Not reported | Yes, semilocal-difference calculation | Not reported |
| `C1`, `C2`, Codex training-refitted references | Yes | Not reported | Not reported | Not reported |

Additional archived controls are under `functionals/`. Every `.py` has a paired `.params.json`; **always use both**. Candidate `init_params()` generally contains a historical initialization, not the reported fitted coefficients. `GAS22` is the published functional; `O2` and `C2` are distinct training refits.

MGCDB84 includes 4,985 reactions: 870 train, 2,963 validation, 1,152 test, using 5,930 unique molecular geometries. Missing-geometry reaction XB51_15 is excluded. The paper's local split is not the original literature's 870/2,960/1,150 split. `datasets/mgcdb84_reactions_syfes.csv` fixes names, stoichiometries, weights, and membership. The optional SAFS26-like reporting partition is separately archived in `mgcdb84_reactions_agentic2026.csv`; it moves the entire C20C24 subset and is only an approximation to the published partition.

GSCDB membership is fixed by [datasets/gscdb36.txt](datasets/gscdb36.txt): 2,734 observations and 3,487 molecules; every selected **dataset's largest molecule** has at most 500 basis functions. This is not the full GSCDB137 and not a filter that discards large molecules within each dataset. Report both 36 sets and 35 excluding AE11. There are 20 new and 16 MGCDB84-derived sets in the 36-set scope.

## Environment

The recorded environment was Linux, Python 3.12.14, PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 14.2.0, CUDA-enabled PyTorch 2.14.1, and MCP 1.30.0. The machine had three RTX 6000 Ada GPUs (48 GB each), 64 CPU cores, and approximately 503 GB RAM. Requirements pin the observed scientific packages; this is not a complete transitive lock or a tested fresh installation on every platform.

From this repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-gpu.txt
python -m pip check
python tools/verify.py
python tools/check_functionals.py
python tools/check_mcp.py
```

Use a CUDA-13-compatible NVIDIA driver. The original driver was 580.178.04. Installation above concerns environment-local wheels; it does not install a driver or system CUDA toolkit. CPU-only saved-energy verification needs only Python's standard library; synthetic functional checks need NumPy, SciPy, h5py and PyTorch. The molecular runners are GPU-oriented: CPU PySCF alone is not a drop-in replacement. Even the AE11 CPU quadrature fallback imports the GPU adapter.

If these exact historical wheels are unavailable on a reader's platform, preserve the recorded versions and document any replacement environment as a new numerical validation. Do not silently relax the pins and claim identical reproduction.

## Acquire the external benchmark inputs

The checkout defaults to `work/data` for MGCDB densities/caches and `work/gscdb` for GSCDB. Set absolute environment paths to use a larger disk; use the same values in the MCP client environment.

```bash
export DFTSR_DATA_ROOT="$PWD/work/data"
export GSCDB_WORK="$PWD/work/gscdb"
export GSCDB_ROOT="$GSCDB_WORK/GSCDB"
export GSCDB_REF_DIR="$GSCDB_WORK/reference"
export GSCDB_RUNS="$GSCDB_WORK/runs"
mkdir -p "$DFTSR_DATA_ROOT/external" "$GSCDB_WORK"
git clone https://github.com/peverati/ACCDB.git "$DFTSR_DATA_ROOT/external/ACCDB"
git -C "$DFTSR_DATA_ROOT/external/ACCDB" checkout bd1f68947daab2370edccd4339cc0390dc562c29
git clone https://github.com/JiashuLiang/GSCDB.git "$GSCDB_ROOT"
git -C "$GSCDB_ROOT" checkout 0624edaad9a694735a13cd5493e2bb63a27e0fd7
```

After the GSCDB checkout, `python tools/check_gscdb.py` recomputes all four saved external jobs on the CPU and checks every reported per-set NER.

The commits above were read from the original local checkouts. The final GSCDB source is **JiashuLiang/GSCDB**, not the older GSCDB138 checkout mentioned in early setup notes. Required files include ACCDB's `Databases/MGCDB84/DatasetEval.csv` and `Geometries/`, and GSCDB's `Allmols_info.json`, `Info/`, `Analysis/Molecule_Energies.csv`, and Q-Chem input files. Follow those upstream projects' citation and license terms. The port embeds the SyFES split/weight definitions and does not require JAX or a google-research checkout.

## Generate MGCDB84 reference densities and caches

Reference protocol: ωB97M-V/def2-QZVPPD, grid level 3, SG-1 VV10 grid, RI-JK orbital optimization and exact-integral energy evaluation. The density files retain reference orbitals, density descriptors, and energy components. This preparation is **SCF even when subsequent candidate evaluation is non-SCF**.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/harness.py opus compute_densities.py --splits train,validation,test
python tools/harness.py opus density_progress.py
python tools/harness.py opus compute_constraint_data.py
python tools/harness.py opus build_feature_cache.py --splits train,validation,test
```

To use several GPUs, launch the density command in separate terminals with `CUDA_VISIBLE_DEVICES=0`, `1`, and `2`. Workers coordinate through molecule lock files and skip completed files. Confirm progress and errors before building caches. Do not use `--allow-partial` for a paper comparison. Constraint preparation computes O₂, H₂O, and AE18 data on the CPU. Caches store float32 grid descriptors; reporting arithmetic is float64. Full-grid evaluation uses the HDF5 density arrays directly.

These steps create large files. Plan for substantial SSD space and host memory; the original setup used a 1.9-TB scratch drive. The package does not estimate peak requirements for every machine. One 48-GB GPU can run the supplied non-SCF evaluator; the original warm worker held roughly 14–16 GB for train/validation, with additional requirements for test and compilation.

## Frozen MGCDB84 non-SCF evaluation

Commands below include test **only for fixed-parameter reader reporting**, after the source/parameter pair is fixed. Without `--include-test`, only train/validation are evaluated. They never refit coefficients or select candidates using test outcomes.

```bash
python tools/experiment.py mgcdb --method O159 --mode nonscf --gpus 0 --include-test --out work/results/O159_fullgrid
python tools/experiment.py mgcdb --method GAS22 --mode nonscf --gpus 0 --include-test --out work/results/GAS22_fullgrid
python tools/experiment.py mgcdb --method O396 --mode cached --gpus 0 --include-test --out work/results/O396_cached
python tools/experiment.py mgcdb --method C356 --mode cached --gpus 0 --include-test --out work/results/C356_cached
python tools/experiment.py mgcdb --method O178 --mode cached --gpus 0 --include-test --out work/results/O178_cached
python tools/experiment.py mgcdb --method C2 --mode cached --gpus 0 --include-test --out work/results/C2_cached
```

Use `--mode cached` for search-metric reproduction and `--mode nonscf` for direct full-grid integration. The two have slightly different precision and must be labeled. `result.json` reports train/validation/test WRMSDs in kcal/mol, with per-category results where supplied by the underlying evaluator. Full-grid outputs also include resumable per-molecule `molecules.jsonl`. WRMSD is `sqrt(mean(weight * error_kcal**2))`, **not** normalized by the sum of weights. The combined validation/test score excludes RG10; it is not an average of the two split WRMSDs.

Expected rounded validation/test pairs: O159 full-grid 3.332/3.818; published GAS22 full-grid 3.462/3.599; O396 cached 3.138/3.841; C356 cached 3.363/3.897; O178 cached 3.481/3.865; C2 cached 3.489/3.628. Exact cached reference checks can differ slightly between the two recorded harnesses.

## Frozen MGCDB84 SCF evaluation

```bash
python tools/experiment.py mgcdb --method O159 --mode scf --gpus 0,1,2 --include-test --out work/results/O159_scf
python tools/experiment.py mgcdb --method GAS22 --mode scf --gpus 0,1,2 --include-test --out work/results/GAS22_scf
```

This evaluates all three splits, starting candidate SCF from the stored ωB97M-V orbitals and retaining both non-SCF and SCF energies. Omit `--include-test` for train+validation. Outputs include `molecules.jsonl`, `result.json`, and `split_report.json`. Use the reference energies from the density campaign for the primary ωB97M-V SCF comparison; a separate Torch ωB97M-V SCF is an implementation check.

Expected validation/test: ωB97M-V reference 3.533/4.309; O159 SCF 3.330/3.805; GAS22 SCF 3.470/3.589. The historical O159 MGCDB campaign converged completely; GAS22 retained 20 unconverged records. Excluding affected reactions changes the split WRMSDs by at most 0.011 kcal/mol. Always report convergence counts and the number of scored reactions. A runner finishing is not evidence that every SCF converged.

For a smaller implementation probe before the full campaign:

```bash
python tools/harness.py opus scf_evaluate.py --candidate "$PWD/functionals/O159.py" --params "$PWD/functionals/O159.params.json" --subset probe --gpus 0 --out "$PWD/work/results/O159_probe"
```

## GSCDB36 non-SCF and SCF evaluation

GSCDB uses its own Q-Chem-specified bases, ECPs, grids, fields, multiplicities, and state searches, rather than the MGCDB def2-QZVPPD protocol. Its ωB97M-V reference orbitals are generated on demand and shared across candidates; they cannot be replaced with MGCDB density files. A first `--mode nonscf` GSCDB run therefore still computes reference SCFs if the orbitals do not exist.

```bash
python tools/experiment.py gscdb --method O159 --mode scf --gpus 0,1,2 --out work/results/gscdb_O159_scf
python tools/experiment.py gscdb --method GAS22 --mode scf --gpus 0,1,2 --out work/results/gscdb_GAS22_scf
python tools/experiment.py gscdb --method O396 --mode nonscf --gpus 0,1,2 --out work/results/gscdb_O396_nonscf
python tools/experiment.py gscdb --method O178 --mode nonscf --gpus 0,1,2 --out work/results/gscdb_O178_nonscf
```

SCF mode retains the matching non-SCF energy, so it reproduces both columns for O159/GAS22. For non-SCF alone, choose `--mode nonscf` with either method. O178 is evaluated directly here; the historical calculation used the equivalent semilocal energy difference from O159. Its archived 36-set mean NER is 1.18915959.

The wrapper excludes AE11_Rn from GPU work, computes its CPU fallback, merges the record, and regenerates `result.json` and `energies.csv`. The CPU fallback is essential for the reported 36-set result. Do not call the bare GSCDB runner on AE11_Rn with GPU4PySCF 1.8.1. The bundled adapter retains the h-shell core-Hamiltonian and exchange workarounds. Inspect `result.json` after the wrapper completes; the underlying `progress.json` predates the final CPU merge.

Expected 36-set mean NER: local reference 1.144; O159 non-SCF/SCF 1.141/1.139; O396 non-SCF 1.147; published GAS22 non-SCF/SCF 1.212/1.190. NER is the published dataset-specific metric divided by that dataset's standard error, then averaged **equally over datasets**. It is not a pooled molecular MAE. `result.json` preserves per-set values so the 35-set, new20, and overlap16 scopes can be computed explicitly.

Historical convergence repair classified energy-stationary/noise-limited records separately from strict convergence. Two GAS22 Pd records remain unconverged. Fresh runs may need similar diagnosis; the wrapper does not automatically apply historical job-specific repairs or label unconverged results as converged. See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

To extend to **all GSCDB137 datasets**, the underlying runner accepts `--subset all`; this is an unreported, substantially more expensive experiment. Use `tools/experiment.py gscdb ... --scope all` to retain the same radon handling. The paper offers no full-GSCDB accuracy claim or completed full-GSCDB candidate results. Neither C356 SCF nor its external performance was reported; commands applied to it create new results.

On the original hardware, a GSCDB36 non-SCF candidate took about 7 GPU-hours after reference preparation; candidate SCF took about 51 GPU-hours for O159 or 65 for GAS22. Reference preparation and difficult electronic states add cost. Use separate output directories for each source/parameter/mode/scope; resume only the identical experiment. The wrapper checks this identity. After diagnosing a failed GSCDB molecule, add `--retry-errors` to retry its failed record; this does not refit the functional.

## Agent harnesses and non-LLM comparison

See [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for MCP registration, seeds, four-island coordination, the evaluation queue, validation-only selection, test access, and SCF tools. The named historical backbones were Opus 5.5 through Claude Code and GPT-6 Astra through Codex. Their availability, accounts, and authentication are external prerequisites; no credentials or model weights are included.

The latest manuscript also reports a 400-trained-form SyFES port. After generating the MGCDB caches/constraints:

```bash
python tools/harness.py opus syfes_run.py --out "$PWD/work/syfes/new_run" --n-forms 400 --gpus 0,1,2 --seed 0 --population 100 --tournament 25 --trials 5 --maxfevals 100000
CUDA_VISIBLE_DEVICES=0 python tools/harness.py opus syfes_recheck.py --out "$PWD/work/syfes/new_run" --top 10
```

This is a PyTorch port of the Ma et al. SyFES workflow, seeded from published GAS22, with fingerprint-equivalence filtering and CMA-ES training. It is not an execution of the original JAX project. Asynchronous worker ordering and stochastic fitting can change the trajectory. Archived `results/syfes/` contains the actual history and recheck: 400 trained forms, 368 finite, 32 nonfinite, best screened validation 3.506, and no novel-form test evaluation.

## Repository contents and release status

- `harnesses/opus`, `harnesses/codex`: distinct scientific implementations, MCP servers, prompts, and preparation/running scripts.
- `functionals/`: evaluated source and frozen coefficients, with ID/metric provenance.
- `datasets/`: exact reporting membership and weighted reaction definitions.
- `results/`: saved molecular energies, external run reports, iteration ledgers, SyFES and later controls.
- `tools/`: portable entry points and verification.
- `archive/`: historical job-specific convergence scripts, retained as evidence rather than turnkey commands.
- `provenance/`: the three project records and copy hashes. These are historical records, not current instructions to readers or agents.

This is a local GitHub-preparation folder; assembling it does not create or publish a remote repository. The manuscript currently points to `BaratiLab/NAMD_AGENT`; verify the intended repository name and link when publishing. No new license has been assigned to the authors' code. Before public distribution, the rights holder should supply its chosen license and publication citation; upstream materials keep their own terms. This package supplies reproducibility materials, not a legal determination of reader entitlements.
