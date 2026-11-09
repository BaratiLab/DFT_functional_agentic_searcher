# SCF evaluation of the lead functional (it159): probe, then a full MGCDB84 run

Context: read `runs/nonscf_campaign_2026-10-15/SUMMARY.md` first. The 200-iteration non-SCF search is finished,
and the lead constraint-clean functional is **iteration 159** (13 parameters, non-SCF val 3.332, −5.7 % vs ωB97M-V).
The user has approved exactly two SCF jobs on it159, run **one after the other**:

1. **Probe:** `scf_submit(159, subset='probe')`. About 60 small validation molecules, roughly 5–15 min.
2. **Full run:** `scf_submit(159, subset='all')`. All of MGCDB84 (train + validation + test), roughly 24 h on 3 GPUs.
   Start it only after the probe has finished and looks sane (step 3).
   **Do NOT run `subset='validation'` separately**; the user skipped it.

Note: the full run includes the test split. The user authorised this for final reporting only. Do not use any
test or SCF result to change the functional, its parameters, or the shortlist.

## Steps
0. Load the MCP tools via ToolSearch: `scf_submit`, `scf_status`, `scf_check_eligibility`, `get_token_usage`,
   `worker_status`, `add_memory`.
   - Call `get_token_usage` and record the cumulative token count as the probe's start value.
   - Record the wall-clock start time.
   - Check `scf_check_eligibility(159)`; it should return eligible.
   - GPUs: the non-SCF worker pool occupies GPUs 0–2 at about 14 GB each out of 48 GB. Check `nvidia-smi`.
     Pass `gpus="0,1,2"` for both jobs if memory allows; otherwise use the default.
1. **Probe.**
   - Submit `scf_submit(159, subset='probe', gpus=...)` and note the job_id and outdir.
   - Poll `scf_status(job_id)` sparingly. Use the Monitor tool or a short background wait loop, not rapid polling.
   - When it finishes, record:
     - start/end time and wall seconds; GPUs used;
     - token delta: `get_token_usage` now minus the start value;
     - molecules done and converged; convergence failures, with names;
     - SCF vs non-SCF WRMSD on the probe reactions, and per data type if available;
     - density-driven error: mean and max |E_SCF − E_nonSCF| per molecule, in kcal/mol;
     - whether the ranking against ωB97M-V is kept.
   - Write everything to `runs/scf_it159/probe_job<ID>.md` and a machine-readable `probe_job<ID>.json`
     (include the raw `scf_status` output).
   - Add a short `add_memory(kind='note')` entry.
2. **Sanity gate.** Do not start the full run, and report to the user instead, if any of these hold:
   - more than 10 % of probe molecules fail to converge;
   - the probe SCF WRMSD is more than 10 % worse than its non-SCF value;
   - there are NaNs or crashes.
3. **Full run.**
   - Record the start tokens and time again.
   - Submit `scf_submit(159, subset='all', gpus=...)`. It is detached and will survive this session.
   - Immediately write `runs/scf_it159/full_job<ID>.md` with: job_id, outdir, command, GPUs, start time, start tokens,
     and the expected duration (~24 h).
   - Set up a detached background watcher that does not depend on this Claude session. For example, a small
     `nohup` bash or Python loop that every 10 min reads `<outdir>/progress.json` and the `scf_jobs` table in
     `runs/iterations.sqlite`. When the job finishes or fails, it appends the final results to
     `runs/scf_it159/full_job<ID>_final.json`:
     - end time and wall hours;
     - molecules converged and failed;
     - SCF and non-SCF WRMSD for train, val, test, and val+test without RG10, plus per-type numbers if available;
     - density-driven error statistics.
   - Log the full run separately from the probe.
4. Report to the user: probe results, the confirmed full-run launch (job id, ETA, log locations), and how to check
   progress (`scf_status(<id>)` or the `final.json`).
   - Token and time accounting must be separate for the probe and the full run.
   - Keep context lean: never paste full molecule lists; summarise.

## Integrity
- SCF and test results are for reporting only. No refits, no new candidates based on them.
- Report the numbers exactly as produced, including failures.
- For the paper, a later step should add references in the same pipeline: ωB97M-V (whose SCF densities are the
  stored references) and GAS22 with published parameters (SCF via `scripts/scf_evaluate.py` directly, because
  GAS22 is not gate-eligible). Ask the user before running those.
