# Provenance, numerical qualifications, and validation

## Sources and assembly boundary

This folder was assembled on 2026-11-09 from the paper in `DFT_functional_agentic_search`, the Opus project's `project_master_details.md`, the Codex project's `project_details.md`, the GSCDB project's `project_details.md`, their corresponding source trees, and the existing paper reproducibility archive. Only this new folder was written. No search, density campaign, molecular SCF campaign, or agent session was launched during assembly.

`provenance/source_manifest.json` records original and packaged SHA-256 hashes for copied artifacts. Original absolute paths identify provenance and are not runtime prerequisites. Frozen candidate files were exported from the read-only iteration databases' evaluated `code` field, and their coefficients from `params_json`; their per-candidate metadata identifies the database/iteration. These evaluated coefficients take precedence over `init_params()`.

The portable layer adds `tools/runtime.py`, `harness.py`, `experiment.py`, configuration generation, and verification scripts. Changes inside copied harness files are limited to:

- Portable default data paths in `config.py` and the GSCDB configuration module.
- Portable import/default paths in the AE11 CPU fallback.
- An explicit `--exclude-molecules` argument in the GSCDB GPU runner so the wrapper can route AE11_Rn to the CPU.
- Active Python/current-directory handling for the SyFES shell launcher and Codex MCP queue/session helpers; GPU selection in the queue comes from its environment.

The full-grid non-SCF entry point uses the same energy bookkeeping as the existing SCF driver: reference total energy plus the exact-exchange difference plus candidate-minus-reference semilocal energy on the frozen reference density. The GSCDB wrapper runs the original driver, completes the CPU fallback, then calls the existing finalizer. Candidate equations, fitted coefficients, dataset weighting, and SCF thresholds were not changed for packaging.

The source trees were copied at assembly time, not reconstructed separately for every historical iteration. Historical equations and exact reported energies remain archived. The `tools/assemble.py` file records the local assembly procedure; it is not a setup command for readers and requires the original author's workspaces.

## Numerical limitations that matter for reproduction

**Fitting versus evaluation.** Never obtain a paper result by calling the plain worker CLI on a candidate and allowing its default fit. Even `--nofit` alone uses `init_params`, which may be an initialization. The reader wrapper explicitly loads the paired frozen coefficients and disables fitting.

**Precision.** Cached screening uses float32 stored descriptors with float64 reported energy arithmetic; full-grid runs use the HDF5 arrays. The primary paper table labels these modes. Small cache/full-grid differences are expected. Stochastic multi-start optimization, asynchronous evaluation order, BLAS/GPU reductions, driver versions, and client/model revisions prevent guaranteed trajectory replay.

**Exchange attenuation.** The Opus implementation retains `DFTSR_RSH_SERIES=legacy|corrected`, default `legacy`, for the large-a low-density expansion. The historical branch contained an incorrect asymptotic series. The paper's analysis showed that correcting it changes no reported rounded result; archived re-evaluations are under `results/attenuation/`. Keep `legacy` for historical comparisons and label corrected-kernel experiments explicitly. The Codex snapshot may not expose the same switch; do not imply that setting it changes every harness. This packaging does not silently alter the scientific kernel.

**GSCDB implementation.** The source preserves CPU core-Hamiltonian construction for h-shell cases, separate full-range/long-range exchange instead of the defective fused GPU path, true-Z VV10 grid handling, electronic-state searches, and CPU treatment of AE11_Rn. The reference and candidate share the GSCDB-specific setup. Matching a functional name alone is insufficient to reproduce its energies.

**Convergence.** Historical GSCDB candidate repair used strict, strict-gradient, noise-limited, and unconverged classes. Noise-limited records had a best gradient below 1e-3 and a small spread among the five smallest-gradient iterates; actual accepted maximum spreads were about 8.3e-8 Eh. These records are not all strict 1e-6-gradient convergence. Two GAS22 Pd records remained unconverged. Raw fresh runs can recover the pre-repair 15 O159/66 GAS22 unconverged records rather than the post-repair counts. The archived repair changed energies by at most about 3.9e-4 kcal/mol and mean NER by at most 1e-5. The saved molecular records preserve the repair labels.

The files under `archive/convergence_fix/` are historical, job-specific code and contain original paths. They are not executed by the portable wrapper. A new campaign should inspect its own unconverged records, reproduce the documented criteria if needed, retain both pre- and post-repair results, and report any changes. Never simply set their `converged` flags to true. Exact archived numerical reporting is available without rerunning these repair jobs.

**External scope.** Use explicit set lists, equal-set NER averaging, and paired per-set comparisons. AE11 can materially change the GAS22 comparison. GSCDB was reporting-only: no parameter fitting, candidate selection, or hyperparameter tuning on that dataset. Larger GSCDB runs, C356 external/SCF runs, and O396 SCF runs are new experiments.

**Information flow.** The second Opus campaign had access to notes containing earlier held-out/SCF outcomes and a train-plus-validation residual scan. It is not equivalent to the first campaign or the Codex clean-room protocol. The archived prompt and project records preserve these distinctions. Historical token ledgers do not establish a controlled model-efficiency comparison.

## Validation performed during packaging

The portable commands were checked syntactically. All 29 frozen functionals passed CPU synthetic-density checks with their fitted parameter shapes, finite energy densities, and the source smoke tests. `tools/verify.py` recomputes the MGCDB table from archived absolute molecular energies using only the standard library, checks complete split counts (870/2963/1152), and verifies copied-file hashes.

Both MCP servers initialized with GPU prewarming disabled and returned their tool schemas (29 Opus tools, 16 Codex tools). CPU reaggregation of all four archived GSCDB jobs matched every saved per-set NER exactly, with 3,487 molecules and 36 complete sets in each job. The source hashes of all 126 copied original artifacts were rechecked and remained unchanged.

The existing scientific environment was used for CPU checks; no clean-network dependency installation was attempted. A fresh GPU molecular campaign and a new paid agent search were not run as part of packaging. Passing syntax, MCP discovery, synthetic-density checks, and saved-energy aggregation validates the package's assembly and entry points, not new molecular calculations. See `provenance/validation.json` for the recorded checks and their scope.
