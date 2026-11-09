# Attenuation large-a branch: corrected-kernel re-evaluation (2026-11-07)

See project_master_details.md §13o for the full account. Run order:

1. `reevaluate.py [cuda:N]`: MGCDB84 non-SCF and SCF (first order) WRMSDs, legacy vs corrected
   → `reevaluation.json`, `per_molecule_shifts.json`, `reevaluate.log`
2. `refit_o159.py [cuda:N]`: 4 cold fits + 1 warm-start fit per kernel → `refit_o159.json`, `refit_o159.log`
3. `gscdb_delta.py GPU SHARD NSHARD` (×3) and `gscdb_delta_rn.py`: per-molecule GSCDB137 shifts → `gscdb_delta/`
4. `gscdb_summary.py`: GSCDB137 metrics with the shifts applied → `gscdb_summary.json`

The kernel switch is `DFTSR_RSH_SERIES=legacy|corrected` (default `legacy`), read in `dftsr/xc_torch.py`.
