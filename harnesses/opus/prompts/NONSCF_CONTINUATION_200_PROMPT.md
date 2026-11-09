# Agentic XC-functional discovery: non-SCF continuation campaign (iterations 201–400)

*Written 2026-10-19 by the lead agent (Claude Code, Opus 5.5) before starting the run, for the record.
Predecessor: `prompts/NONSCF_SEARCH_PROMPT.md` (iterations 1–200, campaign of 2026-10-15, summarised in
`runs/nonscf_campaign_2026-10-15/SUMMARY.md`).*

## 0. Task

Run **200 more non-SCF iterations** of the same evolutionary Plan → Execute → Summarize search, **building on
the best results of the first 200**. Everything in `prompts/NONSCF_SEARCH_PROMPT.md` §1–§7 (fixed frame, data,
score, constraints, candidate API, implementation rules, protocol, integrity rules, desirable outcomes, results
checklist, reporting) applies unchanged unless this file says otherwise. Read that file first.

```
RUN_CONFIG
  budget_iterations : 200 new non-seed records (DB ids 201–400, errors included)
  split_variant     : syfes (unchanged; ωB97M-V published val 3.534, SCF gate 3.428)
  knowledge_mode    : warm  (campaign-1 DB, MCP global memory and SUMMARY.md may be used)
  gpus              : 0,1,2 (one MCP worker per GPU → at most 3 evaluate_functional calls in flight)
  islands           : E, F, G, H (new labels, so campaign-2 statistics stay separable from A–D)
  tags              : every campaign-2 iteration carries the tag `c2`
  harness / model   : Claude Code, Opus 5.5 (claude-opus-5-5), single lead agent issuing parallel calls
```

## 1. Starting point (what campaign 1 established)

Lead frame, all 4/4 constraints, every member SCF-eligible:

* **Exchange:** published ωB97M-V form, g_x = 0.85 + c_w·w + c_u·u_x (γ_x = 0.004).
* **Opposite-spin:** ωB97M-V polynomial with c_os,00 = 1: 1 + c1·w + c2·w² + c3·w⁶ + c4·w²u + c5·w⁶u (γ_os = 0.006,
  spin-averaged w and x²).
* **Same-spin:** GAS22 same-spin polynomial g_ss = u + c1·w + c2·w² + c3·u⁶ + c4·u⁶w⁴ (γ_ss = 0.469) multiplied
  by the valence-window factor W = 1 + (β0 + β2·w_σ²)·r_s,σ/(1 + r_s,σ^b).

| it | variant | params | val | range over refits | gap |
|---|---|---|---|---|---|
| 159 (lead) | b = 4 | 13 | 3.332 (−5.7 %) | 3.332–3.380 | 0.08 |
| 145 | b = 3 | 13 | 3.333 | 3.333–3.425 | 0.11 |
| 198 | b = 4, os pair tied | 12 | 3.345 | 3.345–3.383 | 0.09 |
| 152 | b = 3, β0 only | 12 | 3.344 | 3.344–3.392 | 0.11 |
| 125 | ωB97M-V same-spin polynomial, b = 3 | 14 | 3.337 | 3.337–3.480 | 0.12 |

Code and fitted parameters: `runs/nonscf_campaign_2026-10-15/itNNN.py` / `itNNN_params.json`.
Remaining weaknesses on **validation** vs ωB97M-V: ID (ISOMERIZATION20; 6.22 vs 5.83), SW49 water–sulfate
clusters, NCD slightly worse; BH, NCEC, RG10, IE better.
Dead ends and insights: `get_memory(kind='dead_end')`, `get_memory(kind='insight')` (MCP global memory ids 1–14).
Do not repeat a listed dead end without a stated new reason.

**Bar to clear:** a new candidate counts as an improvement only if it beats **3.332** on validation by more than
the fit-noise floor (~0.02 kcal/mol), is constraint-clean, and the gain survives a second fit
(`n_restarts=3`, or a refit from a different reasonable initialisation). Report the val range across fits, not
the single best number. At equal accuracy (±0.02), fewer parameters and a smaller train–val gap win.

## 2. Island assignment for campaign 2

Each island starts from campaign-1 parents (any iteration id may be used as `parent_id`). An island's archive =
its own campaign-2 iterations plus the campaign-1 seeds listed for it. Parent selection keeps the 80 % random /
20 % top-3 rule over that archive; migrate the best idea of another island every ~10 iterations per island.

| island | seeds (campaign 1) | direction |
|---|---|---|
| **E** | 159, 145, 152, 198 | Refine the same-spin valence window *physically*: what r_s,σ really proxies (kinetic radius, gradient-corrected radius, Fermi-hole size), smooth saturating profiles, amplitude dependence on descriptors other than w² (not ζ, not α-gates: dead), fewer parameters at equal val. No pure retuning of b or polynomial orders. |
| **F** | 159, 125 | Attack the remaining validation weaknesses (ID/ISOMERIZATION20, SW49, NCD, TCE) with physics that keeps the exchange enhancement close to published ωB97M-V (SCF-friendliness): e.g. correlation terms sensitive to intramolecular dispersion-like/medium-range regions, same-spin gradient regimes, high-gradient tails of g_ss. Diagnose from the parent's per-subset val errors first. |
| **G** | 159, 198, 155 | Compactness and robustness: reduce parameters (target ≤ 11) by tying or removing degenerate directions, physically constrained coefficient relations, exact limits that are *compatible* with the data (the exact same-spin UEG limit is known to be harmful). Also re-examine the opposite-spin channel only through constraints/reparameterisations, not added flexibility (os extensions are a dead end). |
| **H** | 159, 145, 3 (B97M-V) | Structurally different ideas: new bounded descriptors, alternative correlation partitions, one-electron self-correlation freedom done in a way that keeps the window, rational forms with positive denominators, and (as a deliberate experiment) transplanting the full lead frame to the pure meta-GGA family. |

## 3. Protocol changes relative to the original prompt

1. **Session start:** `warm_up_workers`, `get_task_brief`, `search_stats`, `get_memory` (all kinds), and
   `list_iterations(sort_by='score', limit=30, include_plans=True)`; read the code of 159/145/152/198/125.
2. **Parallel rounds:** issue up to 3 `evaluate_functional` calls at once, from different islands, each with its
   own written plan, `island`, `parent_id` and `tags='c2'`. Record a factual `record_summary` for every
   iteration (errors included) before planning the next round on that island.
3. **Initialisation:** start new terms so the parent is reproduced (new coefficients 0 or slightly off zero,
   because L-BFGS stalls at an exact stationary point). Initialise inherited coefficients at the parent's
   fitted values *or* at its init values, and say which; campaign 1 showed refits from fitted values reach
   lower train but higher val (path dependence).
4. **Global memory:** every ~10 campaign-2 iterations add `dead_end`/`insight` notes; every ~25, store the
   results checklist (`kind='note'`) for the best ~3 constraint-clean candidates of both campaigns.
5. **Progress note to the user every ~25 iterations** (best so far, ideas that worked/failed, tokens,
   GPU/agent time), and a final report at iteration 400.

## 4. Integrity and gates (unchanged, restated)

* No frozen-parameter grafting; every kept coefficient is in `init_params` and refit.
* **Test set:** `final_test_evaluation` stays disabled. SCF test-split numbers of it159 and GAS22 exist in the
  MCP notes and `runs/scf_*` (computed after campaign 1, for human reporting). They **must not** influence
  parent choice, the direction of the search or the final ranking. Select on validation only.
* **No SCF calculation** (`scf_submit`) without the user's explicit approval naming the iteration; when an
  iteration passes `scf_check_eligibility`, report it and continue the non-SCF search.
* No antisymmetric spin terms, no steep switches, no density-scale dependence in g_x, no NN blocks in exchange.
* Long jobs outside the MCP server (none planned) go through `systemd-run --user`.

## 5. Deliverables at the end

* `runs/nonscf_continuation_2026-10-19/`: `SUMMARY.md` (same structure as campaign 1: references, shortlist with
  val ranges, parameters, per-type errors, lineage, ablations, history of gains, insights, dead ends, cost),
  `iterations_201-400.csv`, `iterations_201-400_full.jsonl`, `global_memory.json`, and code + fitted
  parameters of the shortlisted candidates.
* Final report to the user: top 5 constraint-clean functionals of both campaigns (formula, parameters,
  per-type errors, constraints, lineage), with an explicit statement of whether campaign 2 beat it159 beyond
  fit noise, and the SCF candidates awaiting approval.
