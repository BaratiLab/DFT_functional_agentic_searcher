# Repeating the agent workflow

A new search requires a reader-controlled Claude Code or Codex account and a compatible agent client with MCP support. Frozen-result reproduction in the README does not require either client. Do not give a clean-room search agent the saved result archive or the paper's fitted candidates; doing so changes the information available to the search.

## Register the appropriate server

After installing the environment, from the repository root:

```bash
source .venv/bin/activate
python tools/mcp_config.py claude
python tools/mcp_config.py codex
```

The first command prints the Claude project `.mcp.json` structure, using the Opus harness. The second prints a Codex `config.toml` MCP stanza, using the independent Codex harness. Place/merge the configuration in the appropriate client settings; these generators do not change client settings. They embed the current Python executable and checkout paths, so regenerate after moving the checkout. Override `DFTSR_GPUS` with the available devices and include the same absolute `DFTSR_DATA_ROOT`/`GSCDB_*` paths used during preparation if they differ from defaults.

`DFTSR_ALLOW_TEST=0` keeps the MCP final-test tool disabled. `DFTSR_PREWARM=0` prevents every separate agent connection from allocating a GPU pool. Keep the client tool timeout at least 600 seconds: the server timeout is 560 seconds and the fitting time budget is 410 seconds. Launch the client in the corresponding `harnesses/opus` or `harnesses/codex` directory. The server uses stdio; do not expose it as a network service.

The copied servers and trackers preserve each project's original behavior. Both harnesses use their own `runs/iterations.sqlite` and `candidates/` by default, while sharing read-only density/cache inputs. Do not set both harnesses' `DFTSR_RUNS_DIR` to the same directory. The historical Codex queue expects its database specifically at `harnesses/codex/runs/iterations.sqlite`, so retain that default when using the queue.

## Prepare the three seed iterations

Generate the reference densities, all necessary feature caches, and constraint data as described in the README. On a fresh harness only:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/harness.py opus establish_baseline.py
CUDA_VISIBLE_DEVICES=0 python tools/harness.py codex establish_baseline.py
```

Each establishes the **published** ωB97M-V baseline/J_target and fits three seed forms (ωB97M-V, GAS22, B97M-V) on training data with two restarts. The searches have distinct fitted seeds; new fits need not exactly match historical ones. This is intentionally different from loading frozen paper parameters. Do not replace the paper coefficient files with newly fitted values.

Verify that `search_stats` contains only the seed records before starting a new clean-room search. A prior live search should be preserved in a separate checkout; do not delete its history or reset its counters to make it appear new.

## Coordinator and four islands

The retained historical prompts are:

| Harness | Prompt under that harness's `prompts/` | Role |
|---|---|---|
| Opus | `NONSCF_SEARCH_PROMPT.md` | Candidate API, physical scope, fitting and integrity rules |
| Opus | `MULTIAGENT_NONSCF_400_PROMPT.md` | General four-island protocol |
| Opus | `NONSCF_CONTINUATION_200_PROMPT.md` | Historically informed second campaign (IDs 201–400) |
| Opus | `SCF_LEAD_PROMPT.md` | Later frozen-candidate SCF stage |
| Codex | `NONSCF_SEARCH_PROMPT.md` | Independent run's base specification |
| Codex | `NONSCF_SEARCH_400_MULTIAGENT_PROMPT.md` | Clean-room 400-success protocol, A100/B120/C100/D80 |

These are preserved historical prompts and include obsolete machine paths, dates, preflight statements and previous run IDs. A reader must substitute their checkout paths and actual current state; assertions such as “done on 2026-10-19” are not setup actions. In a fresh reproduction, treat the continuation prompt as a separately identified, history-informed experiment, not an independent clean-room run. The original Opus run had 397 non-seed evaluations across two campaigns; the Codex run had 400 successful non-seed evaluations plus one failed attempt. Do not equate those accounting conventions.

The coordinator assigns four scientific islands, retains a validation-only elite archive, schedules checkpoints, and shares memory. Island agents propose mathematical forms and initialization choices. One evaluation owner submits source through `evaluate_functional`; the tool fits only the training split, evaluates validation, checks constraints, and records source, parent, rationale, coefficients, metrics, and token charges. After receiving a result, the agent calls `record_summary` and updates structured shared memory with `add_memory`. Useful read tools include `get_task_brief`, `list_iterations`, `get_iteration`, `get_lineage`, `get_memory`, `search_stats`, and `get_token_usage`.

Scoring is `J_target / validation_WRMSD * 0.9**n_violations`. The four physical checks cover spin symmetry, uniform-gas exchange, uniform-coordinate exchange scaling, and grid sensitivity. Numerical finite-value checks supplement them. Passing this suite is not evidence that every exact condition is satisfied. Published GAS22 fails the uniform-gas exchange check; retain it as a clearly labeled comparator.

## Shared Codex evaluation queue

The supplied Codex queue owns one MCP connection with a three-worker concurrency cap. Initialize its local directories, then run it from a separate terminal with the same environment:

```bash
mkdir -p harnesses/codex/runs/eval_queue/{pending,active,results}
DFTSR_GPUS=0,1,2 python tools/harness.py codex run_eval_queue.py
```

Island agents write one JSON request at a time under `harnesses/codex/runs/eval_queue/pending/`. A request uses the actual `evaluate_functional` schema:

```json
{
  "name": "MyCandidate",
  "code": "<complete Python source defining one Functional subclass>",
  "plan": "Physical hypothesis, changed expression, and planned comparison",
  "parent_id": 2,
  "island": "A",
  "n_restarts": 1,
  "max_iter": 300,
  "tags": "new-reader-run"
}
```

The source must implement the documented `dftsr.api.Functional` interface; the string above is a schema illustration, not executable candidate code. Results are written to the matching filename under `results/`. The queue defaults to an initial 50-success checkpoint. After each audit, the coordinator writes the next cumulative ceiling (for example 100) to `runs/eval_queue/checkpoint_limit`. It stops at 400 non-seed successes or 480 total non-seed records. A `runs/eval_queue/pause` file pauses new submissions. Do not open another GPU-owning pool for each island; direct metadata connections can remain un-warmed.

A one-GPU queue is also possible by setting `DFTSR_GPUS=0`; several requests may queue against that one worker. Opus uses its MCP worker pool directly through the coordinator. Client-native agent delegation is an orchestration dependency; these prompts are not a standalone autonomous-search executable.

## Freeze, report, and optionally proceed to SCF

Select candidates on training/validation and the documented constraints and parsimony rules. Export the evaluated source plus `params_json` and hash both **before** exposing test data. Preserve all failed and successful attempts, and keep new results separate from `results/`.

For a new search, the MCP `final_test_evaluation` tool requires restarting its server with `DFTSR_ALLOW_TEST=1`; use it only after freezing the selection. Restore `0` afterwards. Reproducing already-published fixed results with `tools/experiment.py --include-test` is a separate reporting operation; it must not feed back into search. The copied historical project rules about one-time testing describe the original experiment, not a prohibition on readers verifying published frozen results.

The MGCDB MCP tools `scf_check_eligibility`, `scf_submit`, and `scf_status` connect the non-SCF search to candidate SCF. Start with `subset="probe"`, then the intended larger split. The default gate requires a successful constraint-clean candidate with validation error below 0.97 times the local published baseline. This gate is for newly discovered candidates; frozen published comparators such as GAS22 are run explicitly with the CLI.

The Opus server also exposes `gscdb_info`, `gscdb_estimate`, `gscdb_scf_submit`, `gscdb_scf_status`, and `gscdb_report`. GSCDB tools use systemd user units and retain a historical `user_approval` parameter. Before launching a live job through that tool, the operator specifies the candidate and scope; the required literal is `it<ID>` or `reference`. IDs refer to **that live database**, not automatically to O159/C356. For the paper's frozen functionals, prefer the README's `tools/experiment.py` command because it also completes the AE11 CPU fallback. The bare MCP GSCDB submission does not perform that final CPU merge. For non-SCF-only GSCDB runs, use the CLI wrapper's explicit mode; the historical submission tool does not expose a non-SCF argument.

For long runs, a systemd user unit separates the calculation from an editor's process group. For example, with default paths, from the repository root:

```bash
systemd-run --user --unit=dft-reader-o159 --collect --working-directory="$PWD" "$PWD/.venv/bin/python" "$PWD/tools/experiment.py" mgcdb --method O159 --mode scf --gpus 0,1,2 --include-test --out "$PWD/work/results/O159_scf"
journalctl --user -u dft-reader-o159 -f
```

If using nondefault data paths, pass them explicitly with systemd `--setenv=NAME=value` options. Systemd does not generally inherit the invoking shell's entire environment. Repeating an identical command resumes completed molecular records; errors and unconverged records require diagnosis, not a new functional fit.

## Token accounting

The Opus tracker reads Claude project transcripts; the Codex tracker reads local Codex session logs for relevant working directories. Log formats and client versions can change, so verify actual attribution on the reader's client before interpreting totals. No private transcripts or credentials are distributed here. Token deltas booked at submission can include concurrent agent work and prior summaries; zero-charge records do not mean zero work. Historical ledgers are measurements of those sessions, not deterministic outputs of a new search or billing estimates.
