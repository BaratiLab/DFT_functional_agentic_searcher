# Multi-agent XC-functional discovery, clean room: 400 non-SCF iterations, beat GAS22 on validation and test separately

You coordinate a team of agents running a long-horizon evolutionary search for a new **range-separated hybrid
meta-GGA** functional, evaluated **non-self-consistently** on fixed ωB97M-V/def2-QZVPPD densities through the
`dft-functional-search` MCP tools. This prompt is built on `prompts/NONSCF_SEARCH_PROMPT.md` (the *base prompt*,
also returned by `get_task_brief`):

* **Unchanged from the base prompt (read them; they are binding):** §1 (fixed vs designed parts, `family='rsh'`),
  §2 (data, fitting, score, the five constraints), §3 (candidate code API and implementation rules), §5 (integrity
  rules).
* **Replaced by this prompt:** §4 (protocol → multi-agent protocol, §5 below), §6 (goals → GAS22 objective, §3 below),
  §7 (reporting, §7–§8 below), and the test-set handling at the very end of the run (§8 below).
* Where the two prompts conflict, this one wins. Where this prompt is silent, the base prompt applies.

---

## 0. Operator pre-flight (the human does this before kickoff; agents only verify it)

**Done on 2026-10-19:** the previous 200-iteration run was moved out of the project to
`/media/ash/PortableSSD/dft_functional_sr_codex_archives/`. That archive holds `candidates/`, `reports/`, `runs/`
(the old DB with 203 iterations and 32 memory entries, the island contexts and the eval queue), and
`scripts/{run_b_island,run_cd_islands,claim_eval_requests,export_iteration_reasoning,finalize_search_artifacts}.py`.
The project now has an empty `candidates/`, an empty `runs/eval_queue/`, and a fresh `runs/iterations.sqlite`. The
fresh DB holds the baseline meta and seed iterations 1–3 only, with no memory and no SCF jobs; the next id is 4 and
the token counter was reset at 2026-10-19 12:03.

**Also done on 2026-10-19:**
* All 7 project MCP server processes from Oct 17–19 (children of the Codex app-server daemon) were stopped. Any
  `dft_functional_sr_codex/mcp_server/server.py` process you see now was started by the current session; that is
  expected and is not a blocker.
* `~/.codex/config.toml` → `[mcp_servers.dft-functional-search.env]` now sets
  `DFTSR_SCF_RUNS_DIR = "/media/ash/PortableSSD/dft_sr/scf_runs_codex_run2"` and keeps `DFTSR_ALLOW_TEST = "0"`. The
  directory does not need to exist yet: no SCF runs in this stage.
* `scripts/run_eval_queue.py` stops at `TARGET_OK = 400` non-seed `ok` iterations or `MAX_RECORDS = 480` non-seed
  records, whichever comes first.

Remaining human step: start the agent from a **new** Codex session (no resumed transcript, no carried-over agent
memory).

Verification is the agent's job: check the items above (§1). If they hold, proceed without asking. Stop only for a
real failure, such as a non-seed iteration in the DB, a non-empty memory, files from the previous run, a project MCP
server started before this session, or `DFTSR_ALLOW_TEST` other than `0`. GPU processes from other projects (e.g.
`/home/ash/Documents/dft_functional_sr/`) are not blockers. They share the GPUs, so evaluations may be slower.

---

## 1. Clean-room rule: no knowledge of the previous run

This run must be independent of every earlier search in this project. **You may not see, use, or reconstruct any
result, plan, code, summary, memory or report produced by a previous iteration of the search defined by
`prompts/NONSCF_SEARCH_PROMPT.md`.** This applies to you and to every sub-agent you start. Copy this section verbatim
into each sub-agent's instructions.

**Forbidden sources** (do not open, list, grep, glob, `cat`, or ask a sub-agent to summarise them):
* `/media/ash/PortableSSD/dft_functional_sr_codex_archives/` (everything under it),
  `/media/ash/PortableSSD/dft_sr/scf_runs_codex/`, and any other `*_archives` or `archive` folder.
* Any `*.sqlite` other than the live DB behind the MCP server, and any file with iteration-derived content that
  predates this session (if something like `reports/`, `runs/island_*` or `runs/reasoning_*` reappears, it is not
  from this run).
* Old agent transcripts and memories: `~/.codex/sessions/`, `~/.codex/*.sqlite`, `~/.claude/projects/*/` (transcripts
  and `memory/`), and any other conversation history.
* The previous run's controller scripts (archived; do not restore or recreate them from memory).
  `scripts/run_eval_queue.py` is allowed as infrastructure only.
* Recollection: if you believe you "remember" a form, score or dead end from an earlier session of this project, treat
  it as unknown and do not use it.

**Allowed sources:** the base prompt, this prompt, `docs/FUNCTIONAL_FORMS.md` (literature only), `baselines/*.py`,
the PDFs in the project root (published literature), the `dftsr/` library source, and **only** iterations, summaries
and memory entries created in this run (ids ≥ 4 in the fresh DB).

**Start-of-session verification (the coordinator does this before anything else):**
1. `search_stats` and `list_iterations(sort_by='id', limit=50, status=None)` must show **only** seed iterations 1–3
   (ωB97M-V refit, GAS22 refit, B97M-V refit) and no other ids.
2. `get_memory()` must be empty.
3. `candidates/` must be empty, and `reports/`, `runs/island_*` must not exist.

If any check fails, **stop, change nothing, and report to the user** which artifact is still visible. Do not "work
around" it by ignoring the data; the run would no longer be clean.

If a previous-run fact reaches you anyway (e.g. through a tool output), do not use it, record
`add_memory(kind='note', text='LEAK: <what, where>')`, and tell the user in the next progress note.

---

## 2. Physical constraints (hard requirements)

All of base §1, §3 and §5 apply. A candidate is a **winner or shortlist candidate only if it satisfies all of the
following**. Candidates that violate them may be recorded as stepping stones, but they can never be reported as results.

**Checked by the evaluator; all five must pass (`n_violations == 0`):**
1. Spin symmetry of e_xc (O2 triplet grid, < 1e-5 Ha).
2. UEG exchange limit: g_x(s=0, τ=τ_UEG) = 0.85 within 0.5 %. *GAS22 fails this (g_x ≈ 0.862); its constraint-clean
   repair is a legitimate target.*
3. Uniform coordinate scaling of the exchange enhancement factor (H2O grid, < 1e-4 Ha).
4. AE18 grid sensitivity: (99,590) vs (250,974) < 0.015 kcal/mol.
5. Numerical sanity: finite on every grid point.

**Enforced by construction (reviewers reject plans that break these):**
* Fixed physics is untouched: ω = 0.3, 0.15 SR / 1.0 LR exact exchange, VV10 (b = 6, C = 0.01), `family='rsh'`.
  `family='mgga'` is not used in this run.
* g_x depends only on scale-invariant per-spin descriptors (x², s², t, w, α, u-maps of these). It does not depend on ζ,
  ρ, r_s, or anything else with an absolute density scale. Build it as `0.85 + Σ(terms that vanish at x²=0, w=0)` so the
  UEG limit holds **exactly**, not by fitting.
* Correlation keeps Stoll partitioning on PW92. Same-spin terms are symmetric under α↔β (no antisymmetric spin terms).
  Prefer forms that remove one-electron self-correlation (τ_W/τ → 1 ⇒ same-spin correlation → 0).
* Smooth, bounded, finite everywhere: u-type maps, tanh, positive (softplus) denominators, moderate slopes, no steep
  switches, no NN blocks with free biases in exchange, no Laplacian.
* All kept coefficients are in `init_params` and refit on train (no frozen-parameter grafting). Fixed constants are
  limited to genuine physics (0.85, PW92 constants, γ values inherited from ωB97M-V/GAS22).
* Non-SCF realism: keep the exchange enhancement factor close to ωB97M-V/GAS22 in shape, since large exchange changes
  are mis-ranked by density-driven errors and would not survive the later SCF check.

---

## 3. Objective: improve on GAS22 on validation and on test, separately

The reference is the **GAS22 seed, iteration 2** (refit on this split, same pipeline). Read its live numbers with
`get_iteration(2)`. At setup: val ≈ 3.489 kcal/mol, 1 violation (UEG). The secondary reference is ωB97M-V
(published parameters: `baseline_val_wrmsd` from `get_task_brief`, ≈ 3.534; refit seed 1: val ≈ 3.789).

The two improvements are judged **independently**; one does not substitute for the other:

| goal | metric | when it is known | requirement |
|---|---|---|---|
| **V**: beat GAS22 on validation | val WRMSD < GAS22 seed val; report Δ% = 100·(1 − val/val_GAS22) | live, every iteration | 0 violations |
| **T**: beat GAS22 on test | test WRMSD < GAS22 seed test; report Δ% the same way | **only in the final stage (§8)** | same candidate, same frozen parameters |
| stretch | ≥ 2 % on each of V and T | | ≤ 20 trainable params |

**The test split stays sealed for the entire search.** `final_test_evaluation` is disabled (`DFTSR_ALLOW_TEST=0`) and
must not be called, worked around, or approximated during the 400 iterations (no reading test caches, no reconstructing
test reactions from MGCDB84 files, no evaluating on test through `dftsr` directly). You can only improve T **indirectly**,
by choosing candidates that generalise. GAS22's published gain is larger on test (−12.4 %) than on validation (−7.0 %),
so a validation-only edge over GAS22 is unlikely to carry over to test unless it is broad. Use these proxies when
choosing parents and the shortlist:
* **Breadth:** no validation data type (NCED, NCEC, NCD, IE, ID, TCE, TCD, BH, AE18, RG10) more than ~5 % worse than GAS22's.
  Improvements that come from a single type are suspect.
* **Small train–val gap**, no larger than the GAS22 seed's gap. A gap that grows along a lineage is a risk flag.
* **Parsimony:** fewer parameters at equal validation accuracy (base §5: within ~0.02 kcal/mol, fewer params wins).
* **Physics over patchwork:** forms whose gain comes from an identifiable limit or regime.

---

## 4. Budget

* **400 iterations** = 400 non-seed `evaluate_functional` records with `status='ok'`. Failed evaluations do not count
  toward the 400, but each candidate gets at most 2 fix-and-resubmit retries, and the run stops at **480 recorded
  iterations in total** even if fewer than 400 are `ok`.
* **Default allocation:** 4 islands × 100 `ok` iterations.
* **Reallocation (coordinator only):** at a checkpoint (§7), if an island's best val has not improved for its last 30
  iterations, the coordinator may move up to 20 of its remaining budget to the island with the best recent progress.
  The total stays 400. Log every reallocation with `add_memory(kind='note')`.
* **Per-evaluation limits:** ≤ 3 restarts and ≤ 500 L-BFGS iterations (base §4). If an evaluation times out, simplify
  the form instead of retrying it unchanged.
* **No SCF in this run.** Do not call `scf_submit`. `scf_check_eligibility` may be used to list candidates for the final report.

---

## 5. Multi-agent protocol

### 5.1 Roles

**Coordinator (you, the main session).** Owns pre-flight verification (§1), `warm_up_workers`, the global budget
counter, migration scheduling, global memory curation, the audits, checkpoints and the final stage. Starts the
sub-agents below and gives each one: this full prompt, its island label, its budget, its theme, and the concurrency
rules. The coordinator does not leak information to sub-agents beyond this run's DB and memory.

**Island agents A, B, C, D (one sub-agent each, long-lived or restarted per batch of ~10 iterations).** Each one runs
Plan → Execute → Summarize (base §4, steps 1–5) on its own island. All four start from the seeds 1–3. Each island has
a **starting theme** chosen from physics, so the islands diverge early. After 30 iterations an island may leave its
theme if summaries justify it.

| island | starting theme (physics, not prior results) |
|---|---|
| A | **Exchange:** constraint-clean bounded kinetic/gradient descriptors in g_x (e.g. α-based iso-orbital regimes, u-maps of x² and w) with UEG exact by construction. Start with a GAS22-style form whose UEG violation is removed. |
| B | **Same-spin correlation:** one-electron self-correlation freedom (τ_W/τ), iso-orbital vs slowly-varying regimes, bounded rational forms. |
| C | **Opposite-spin correlation:** r_s- and ζ-dependence, high-density/low-density limits, positive-denominator Padé forms. |
| D | **Parsimony:** ≤ 15-parameter forms that match ωB97M-V/GAS22 accuracy, then add one physically motivated term at a time; aimed at test generalisation (small gap, broad per-type gains). |

**Auditor (a fresh-context sub-agent, started at each checkpoint, §7).** Read-only. Gets the current top 10
constraint-clean iterations (code, plan, fitted params, per-type errors, gap) and checks them against §2 and base §5:
frozen-parameter grafting, hidden ζ or density-scale dependence in g_x, steep switches, antisymmetric spin terms,
over-parameterisation, growing train–val gap, single-type gains. It returns a list of flagged ids with reasons. Flagged
iterations are excluded from parent top-3 selection and from the shortlist, unless a later fix is recorded. The auditor
never designs candidates.

### 5.2 Per-island loop (each island agent)

1. **Read context:** `get_memory(kind='dead_end')`, `get_memory(kind='insight')`, then
   `list_iterations(island=<X>, sort_by='score', limit=30, include_plans=True)`. Only this run's data exists.
2. **Select a parent:** ~80 % a random member of the island archive, ~20 % one of the island's top-3 that are
   constraint-clean and not flagged by the auditor. For the first iterations, the seeds are the archive.
   Call `get_iteration` and `get_lineage`.
3. **Migration:** every 10th iteration of an island, take the best constraint-clean idea from the next island in the
   ring A→B→C→D→A and fuse it into the parent. Set `tags='migration'`.
4. **Plan** (written into `plan`, as in base §4.2): the exact mathematical change; the physical rationale and the data
   types expected to move, quoting the parent's per-type errors **and the GAS22 seed's**; how new parameters are
   initialised (reproduce the parent where possible); and which of the five constraints and §2 rules the change could
   threaten, and why it will not.
5. **Execute:** `evaluate_functional(name, code, plan, parent_id, island=<X>, tags=...)`, routed according to §5.3.
6. **Summarize:** `record_summary` with facts only, as in base §4.4, **plus** Δval vs the GAS22 seed (%), the
   per-type deltas vs GAS22, and the train–val gap vs GAS22's gap.
7. **Memory:** every ~10 island iterations, `add_memory(kind='insight' | 'dead_end', text, iteration_ids)`, using only
   evidence from this run. Every `dead_end` needs ≥ 3 failed attempts with a common cause.
8. Report to the coordinator after each batch of ~10: the island's best (id, val, Δ% vs GAS22, violations, params,
   gap), and its count of ok/error iterations.

### 5.3 Concurrency and GPU access

* There are 3 GPU workers. **At most 3 `evaluate_functional` calls may be in flight at any time, across all agents.**
  Queue waiting counts toward the 10-minute MCP call cap, so a 4th concurrent call risks a timeout.
* **If sub-agents share the coordinator's MCP connection** (one server process), island agents call
  `evaluate_functional` directly. The coordinator hands out 3 evaluation slots in rotation: the four islands take
  turns, and an island waits for a slot before it submits.
* **If each sub-agent gets its own MCP server process** (e.g. Codex sub-agents), only one process may own GPU
  workers. Run `scripts/run_eval_queue.py` (with the 400/480 stop condition from §0) as the single evaluator. Island
  agents write `runs/eval_queue/pending/<island><NNN>.json`, containing exactly the `evaluate_functional` arguments,
  and read their result from `runs/eval_queue/results/`. They use their own MCP connection only for read and summary
  tools, and never call `warm_up_workers` or `evaluate_functional`.
* Only the coordinator calls `warm_up_workers`, once, at the start.

---

## 6. Selection rule for "best" (used at checkpoints and for the shortlist)

The ranking is among iterations with `status='ok'`, `n_violations == 0`, and not flagged by the auditor:
1. lowest val WRMSD;
2. ties (within 0.02 kcal/mol): fewer parameters, then smaller train–val gap, then fewer data types worse than GAS22.

The `score` field (base §2) is a useful sort key, but it does not override this rule.

---

## 7. Checkpoints and reporting (every 50 `ok` iterations; 8 in total)

At each checkpoint the coordinator:
1. runs the auditor (§5.1);
2. considers budget reallocation (§4);
3. updates the **results checklist**, stores it with `add_memory(kind='note', text=<table>)`, and gives the user a
   short progress note covering: the best 3 by §6 (id, val, **Δ% vs GAS22 val**, Δ% vs ωB97M-V, params, gap,
   worst type vs GAS22), the structural ideas that worked or failed in each island, any leaks, `get_token_usage`, and
   the `timing` block of `search_stats`.

Checklist (the base §6 table with GAS22-specific rows; test cells stay "sealed" until §8):

| # | criterion | it … | it … | it … |
|---|---|---|---|---|
| 1 | val WRMSD; Δ% vs GAS22 seed; Δ% vs ωB97M-V baseline | | | |
| 2 | **V**: beats GAS22 on val (yes/no; ≥ 2 %?) | | | |
| 3 | **T**: beats GAS22 on test | sealed | sealed | sealed |
| 4 | 5/5 constraints pass (list any failures); auditor flags | | | |
| 5 | trainable parameters (≤ 20 desirable) | | | |
| 6 | train–val gap vs GAS22 seed gap | | | |
| 7 | per-type: types > 5 % worse than GAS22 on val (list) | | | |
| 8 | physical rationale in one sentence | | | |
| 9 | island and lineage depth; migrations involved | | | |
| 10 | cumulative tokens to reach it; % val gain vs GAS22 per M tokens | | | |
| 10b | GPU seconds of its evaluation; cumulative wall time to reach it | | | |
| 11 | SCF / GSCDB137 transfer (later stage, pending) | pending | pending | pending |

---

## 8. Final stage (after the 400th `ok` iteration, or the 480 cap)

1. **Freeze.** No further `evaluate_functional` calls by any agent. Stop the island agents and the queue runner.
2. **Pre-register the shortlist before any test number exists.** Select the top 5 by §6. Write them, with their
   validation numbers and the selection rule, to `add_memory(kind='note', text='PREREGISTERED TEST SHORTLIST: ...')`,
   and show the list to the user. Iterations 1 (ωB97M-V refit) and 2 (GAS22 refit) are always added as references.
3. **Ask the user to unseal the test split:** restart the MCP server with `DFTSR_ALLOW_TEST=1`. Do not edit the config
   or environment yourself. Wait for confirmation.
4. **One test evaluation per shortlisted id**, via `final_test_evaluation(iteration_id)` with the stored, frozen
   parameters: the 5 candidates + iterations 1 and 2. No refits, no re-selection, no new iterations, and no edits to
   the shortlist after test numbers are seen. Then ask the user to set `DFTSR_ALLOW_TEST=0` again.
5. **Report V and T separately**, for every shortlisted candidate:

   | it | params | violations | val | Δ% val vs GAS22 | test | Δ% test vs GAS22 | val+test (no RG10) | Δ% vs ωB97M-V (val / test) | train–val gap | V? | T? |
   |---|---|---|---|---|---|---|---|---|---|---|---|

   Add per-type test errors vs GAS22 for each candidate. Give an honest verdict: which candidates beat GAS22 on V,
   which on T, which on both, and which on neither. If none beats GAS22 on test, say so plainly. Do not re-describe
   the result.
6. **Final report** (base §7), for the top 5: explicit formula, fitted parameters, per-type errors, constraint table,
   lineage, island, and the full checklist with test filled in.
7. **SCF:** list the SCF-eligible candidates (`scf_check_eligibility`) and **ask the user before submitting anything**
   (each SCF run takes about 24 h).

---

## 9. Kickoff message (for the user to paste)

> Run the 400-iteration multi-agent non-SCF search in `prompts/NONSCF_SEARCH_400_MULTIAGENT_PROMPT.md`. First do the
> clean-room verification in §1 and stop if anything from the previous run is visible. Keep the test split sealed until
> §8, pre-register the shortlist, and ask me before unsealing test and before any SCF.
