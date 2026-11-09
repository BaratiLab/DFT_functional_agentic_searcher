# Project details: Codex non-SCF XC-functional search

Results exported: `2026-10-26T22:52:25.038174-04:00`.

This document records the completed 400-trial Codex search and the subsequent authorized evaluation of its best validation candidate on the held-out test set. Metrics are WRMSD in kcal/mol unless otherwise stated. The results below come from the local database, frozen artifacts, audit reports, and saved MCP outputs.

## Outcome

**The search improved validation accuracy, but the tested candidate did not beat GAS22 on test.**

Codex completed **400 successful non-seed non-SCF evaluations**, plus **one preserved failed attempt**. Three seed records bring the database total to **404 records**. All 401 search attempts have a recorded pre-evaluation plan and post-evaluation summary.

Candidate **356**, `B101_compact_OS_endpoint_pair`, was the best raw-validation candidate and the first selection under the prescribed ranking rule. It has **14 trainable parameters** and **zero evaluator violations**. Its validation WRMSD was **3.362805400**, or **3.614835% below GAS22**. Its frozen test WRMSD was **3.897343411**, or **7.435360% above GAS22**. Thus the requested win on all metrics was not achieved.

| Metric | Candidate 356 | GAS22 reference 2 | Interpretation |
|---|---:|---:|---|
| Training WRMSD | 3.207428316 | 3.266109037 | Training fit, not held-out evidence |
| Validation WRMSD | 3.362805400 | 3.488924272 | Candidate is 3.614835% better |
| Test WRMSD | 3.897343411 | 3.627617040 | Candidate is 7.435360% worse |
| Validation + test, excluding RG10 | 3.691519562 | 3.688485684 | Candidate is 0.082253% worse |
| Validation minus training gap | 0.155377084 | 0.222815235 | Smaller gap did not guarantee test transfer |
| Trainable scalar parameters | 14 | 12 | Actual counts in this implementation |
| Search-stage evaluator violations | 0 | 1 | GAS22 fails the specified UEG exchange check |

Parameter counts are the evaluator's actual refitted scalar counts, not necessarily the counts quoted for published versions of the functionals. No SCF calculations or external-dataset transfer calculations were run. Test access is disabled again. The other four shortlisted candidates remain untested.

## Saved results and reasoning

- [Structured project results](project_results.json): all 404 records, their metrics, fitted parameters, constraints, plans, summaries, source hashes, shortlist, references, test outputs, audits, and accounting.
- [Complete iteration-results CSV](project_iteration_results.csv): all 404 records, including seeds, the failed attempt, and every stored plan and summary. Blank test cells mean **untested**, not zero error.
- [Searchable iteration journal](runs/current_session/iteration_reasoning.html) and [JSONL](runs/current_session/iteration_reasoning.jsonl): all 401 search attempts with scientific hypotheses, outcomes, interpretations, and corrections.
- [Per-iteration token splits and concise rationale](runs/current_session/iteration_accounting.md) and [matching CSV](runs/current_session/iteration_accounting.csv): all 400 successes plus the failed attempt, with token components, an abridged original plan, observed outcome, and links to the full record.
- [Detailed validation report](runs/current_session/final_validation_report.html) and [text](runs/current_session/final_validation_report.txt): formulas, fitted coefficients, lineage, per-type errors, audits, and costs. These preserve the validation-stage report with a later test-result notice.
- [Final held-out test report](runs/current_session/test_evaluation_best/report.html), [text](runs/current_session/test_evaluation_best/report.txt), and [JSON summary](runs/current_session/test_evaluation_best/summary.json).
- [Authoritative database](runs/iterations.sqlite), [run manifest](runs/current_session/run_manifest.json), [provenance verification](runs/current_session/final_integrity_check.json), and [final independent audit](runs/current_session/audit_400.json).

This Markdown document is the narrative overview. The JSON, CSV, and journal preserve the individual results and reasoning rather than reducing the experiment to its successful candidates.

## Project and session history

- Workspace: `/home/ash/Documents/dft_functional_sr_codex`.
- Original Codex session: `01a154de-a703-70b0-9806-4d4913cc317c`.
- Governing instructions: [400-trial multi-agent prompt](prompts/NONSCF_SEARCH_400_MULTIAGENT_PROMPT.md), with the relevant fixed-physics, API, fitting, and integrity requirements from the [base prompt](prompts/NONSCF_SEARCH_PROMPT.md).
- The search began on **2026-10-19**, with a manifest start time of **12:10:22 America/New_York**. Search records begin at ID 4; IDs 1–3 are literature seed refits.
- The interrupted session left **215 successful evaluations and one error**, through ID 219. Its island totals were A50/B55/C55/D55. Outcome summaries for records 217–219 were completed when the work resumed.
- On **2026-10-26**, Codex resumed this same experiment and completed the remaining **185 successful evaluations**. This was not a second independent search or a reset of the database.
- The evaluator stopped at **400 successes**, with final allocations **A100/B120/C100/D80**. The search was frozen before any test evaluation.
- The validation report was finalized on October 26 local time; its manifest timestamp is `2026-10-27T00:33:11Z`.
- The user subsequently authorized Codex to restart test access itself and evaluate the best functional. The test run completed at `2026-10-27T02:13:30Z`, corresponding to **October 26, 22:13:30 America/New_York**.

The original protocol required a clean-room start from the three seeds. Resumption deliberately continued the existing run and its current-run memory. Prior independent-search archives were not used as scientific context. Historical 200-trial instructions in the original README setup section do not define this completed 400-trial run.

## Objective, data, and fixed physics

The primary reference was **GAS22 seed 2, refitted on this training split**. The objective was to beat it on validation and, independently, on test while satisfying the specified constraints. The stretch goal was at least **2% improvement on each split**, with at most **20 trainable parameters**. A validation win could not substitute for a test win.

The published-parameter ωB97M-V validation baseline was **3.533590004**. The separately training-refitted ωB97M-V seed 1 is a different reference and has different scores.

| Split | Reactions | Role |
|---|---:|---|
| Training | 870 | Parameter fitting |
| Validation | 2963 | Evolutionary selection and reporting |
| Test | 1152 | Sealed throughout search; final authorized evaluation only |

The dataset was **MGCDB84**, using the **SyFES split**. Candidate energies were evaluated non-self-consistently on fixed **ωB97M-V/def2-QZVPPD densities**:

```text
E_candidate = E_reference
            + E_semilocal,candidate[rho_reference]
            - E_semilocal,reference[rho_reference].
```

Candidate densities were not recomputed self-consistently. All search candidates used `family='rsh'` with this fixed energy frame:

```text
E_xc = E_x,semilocal^SR[g_x]
     + 0.15 E_x,HF^SR(omega=0.3)
     + 1.0 E_x,HF^LR(omega=0.3)
     + E_c,ss[g_ss] + E_c,os[g_os]
     + E_c,VV10(b=6, C=0.01).
```

Codex designed the semilocal enhancement factors and correlation terms. Range separation, exact-exchange fractions, and VV10 parameters stayed fixed. Correlation used the **Stoll partition of PW92**. Exchange depended only on scale-invariant per-spin descriptors, with the UEG exchange value **0.85** imposed by construction.

All retained trainable amplitudes and introduced shape parameters were refitted on training data. Parent fitted coefficients were not frozen as hidden constants. Fixed inherited gamma values, physical constants, algebraic identities, and numerical guards were distinguished from new empirical shape scales.

Shared inputs were under `/media/ash/PortableSSD/dft_sr`, including density, feature-cache, and external-data directories. The configured SCF destination for this run was `scf_runs_codex_run2`; no SCF jobs were submitted.

## Fitting and evaluation

- Training objective: **WRMSD²**, optimized using L-BFGS and autograd.
- Per-evaluation limits: at most **3 restarts** and **500 L-BFGS iterations**.
- Restarts were selected using the **training objective**, not validation or test.
- Server job timeout: **560 seconds**; usual fitting time budget: **410 seconds**.
- The existing PyTorch evaluator performed compiled fitting and float64 reported metrics.
- Internal search score: `J_target / validation_WRMSD * 0.9**n_violations`, with `J_target = 0.858 * published_omegaB97M-V_validation_WRMSD`.
- Final ranking used the explicit validation/parsimony/gap rule below, rather than score alone.

The initial seed records were:

| ID | Reference | Parameters | Training | Validation | Violations |
|---|---|---:|---:|---:|---:|
| 1 | ωB97M-V refit | 13 | 3.191269673 | 3.789175545 | 0 |
| 2 | GAS22 refit | 12 | 3.266109037 | 3.488924272 | 1 |
| 3 | B97M-V refit | 14 | 6.158617447 | 6.273950184 | 0 |

## Codex orchestration and per-iteration reasoning

Codex coordinated four island agents and fresh independent auditors. One shared [evaluation queue](scripts/run_eval_queue.py) owned the GPU-worker connection and enforced at most **three concurrent evaluations**. Island agents submitted source and plans to `runs/eval_queue/pending/`; request and result artifacts were retained in `active/` and `results/`. Metadata, summaries, and memory were handled through MCP, commonly via [session_mcp.py](scripts/session_mcp.py).

| Island | Starting theme | Successful trials | Failed attempts |
|---|---|---:|---:|
| A | Constraint-clean exchange descriptors and forms | 100 | 0 |
| B | Same-spin correlation and orbital response | 120 | 0 |
| C | Opposite-spin correlation and density/spin dependence | 100 | 0 |
| D | Parsimony and compact forms | 80 | 1 |

The initial allocation was 100 successes per island. At checkpoint 200, **20 remaining trials were transferred from D to B**, following D stagnation and B progress. No further budget reallocation was applied. Islands could move beyond their starting themes after 30 trials.

Each iteration followed **Plan → Execute → Summarize**:

1. Read current-run memory, the selected parent, lineage, and training/validation/type errors.
2. Use approximately 80% random archive parents and 20% eligible elite parents. Every tenth local iteration performed the required migration ring: A used B ideas, B used C, C used D, and D used A.
3. Before evaluation, record the mathematical change, physical rationale, affected regimes/types, initialization, parameter budget, and constraint risks.
4. Submit complete source for training-only fitting and non-SCF evaluation.
5. Record observed parent/GAS22 deltas, type changes, constraints, gap, interpretation, and limitations. Preserve failed hypotheses rather than rewriting them as successes.
6. Save batch insights and audit source/parameter consistency. Same-form fitting controls and near-duplicate architectures were labeled explicitly.

Synthetic CPU checks verified formulas and scalar activity without accessing held-out molecular data. Nonzero synthetic sensitivity demonstrates computational activity, not identifiability, optimization convergence, or physical transfer. Because retained parameters were jointly refitted, score changes cannot generally be attributed uniquely to a single modified term.

## Constraints, audits, and ranking

All **400 successful search records reported zero evaluator violations**. This did not mean every record passed the broader construction, realism, parsimony, or generalization review: the final audit retained **29 cumulative exclusions**.

| Evaluator check | Candidate 356 statistic | Recorded threshold |
|---|---:|---:|
| Spin symmetry | 8.4368338e-16 | 1e-5 |
| UEG exchange error, with g_x = 0.85 | 4.5519144e-15 | 0.005 |
| Uniform exchange scaling error | 3.8948813e-15 | 1e-4 |
| AE18 grid sensitivity, kcal/mol | 8.3774966e-5 | 0.015 |
| Numerical sanity | Successful finite evaluation | Finite outputs |

AE18 has no validation observations; its grid-sensitivity pass is not an AE18 validation-accuracy result. NCD, AE18, and RG10 have no observations in this test split.

Fresh audits were conducted every 50 successes. Auditors checked source/plan consistency, active trainable coefficients/scales, fixed physics, symmetry/scaling, bounded smooth factors, exchange-shape realism, and gap/breadth evidence. Synthetic exchange comparisons used physically coupled descriptors, not inconsistent independent extremes. They were not density-weighted or SCF checks.

| Successful evaluations | Preferred top 3 at checkpoint | Saved checklist |
|---:|---|---|
| 50 | 7, 41, 10 | [Checkpoint 50](runs/current_session/checkpoint_050.json) |
| 100 | 61, 98, 100 | [Checkpoint 100](runs/current_session/checkpoint_100.json) |
| 150 | 145, 61, 127 | [Checkpoint 150](runs/current_session/checkpoint_150.json) |
| 200 | 145, 61, 127 | [Checkpoint 200](runs/current_session/checkpoint_200.json) |
| 250 | 61, 204, 145 | [Checkpoint 250](runs/current_session/checkpoint_250.json) |
| 300 | 257, 61, 261 | [Checkpoint 300](runs/current_session/checkpoint_300.json) |
| 350 | 350, 353, 339 | [Checkpoint 350](runs/current_session/checkpoint_350.json) |
| 400 | 356, 395, 384 | [Checkpoint 400](runs/current_session/checkpoint_400.json) |

The exact deterministic ranking rule was:

1. Among successful, constraint-clean, non-excluded records, find the lowest remaining validation score.
2. Consider records within **+0.02 kcal/mol** of that minimum.
3. Prefer fewer parameters, then smaller validation-minus-training gap, then fewer validation types worse than GAS22, then lower validation.
4. Remove that record and repeat.

Pairwise approximate ties are not transitive, so this moving-band order differs from raw validation order. Records were not deduplicated by symbolic form. A roughly 5% per-type regression was a warning, not an automatic exclusion threshold.

For example, Gaussian-screen record 377 was excluded because training error improved while validation worsened, the gap exceeded GAS22's, and category balance deteriorated. Its Gaussian function itself passed construction checks. Record 329 was excluded for added-flexibility regression against nested simpler record 319. Record 395 remained eligible with explicit tradeoffs, because both train and validation worsened versus its parent rather than showing the same train/validation divergence.

## Search development and scientific interpretation

| ID | Island | Parameters | Validation | Gain vs GAS22 | Observed contribution |
|---:|---|---:|---:|---:|---|
| 204 | D | 13 | 3.487481475 | +0.041354% | Small validation edge from a same-form fitting control; not a new symbolic form. |
| 261 | A | 13 | 3.487015928 | +0.054697% | Another small edge from a known-form pruning/fit control. |
| 305 | A | 14 | 3.488511212 | +0.011839% | Density-weighted OS kinetic mean produced a tiny structural validation edge. |
| 319 | A | 13 | 3.455056495 | +0.970723% | Independent OS roots plus a low-gradient kinetic term gave a larger standard-exchange gain. |
| 331 | C | 15 | 3.436026039 | +1.516176% | Migration of density-screened OS response improved raw validation. |
| 339 | B | 15 | 3.406689795 | +2.357015% | Shared SS coefficients plus an orbital window exceeded the 2% validation target; all types within 3% of GAS22. |
| 350 | B | 15 | 3.373554704 | +3.306738% | Quadrature density screen improved aggregate validation; NCED remained weak. |
| 353 | B | 16 | 3.371371247 | +3.369320% | Independent-SS counterpart slightly improved raw validation with an extra parameter. |
| 356 | B | 14 | 3.362805400 | +3.614835% | Pure OS coefficient tie produced the best raw validation with 14 parameters. |
| 384 | A | 14 | 3.373369586 | +3.312043% | Different shared-root compression nearly matched 350 with 14 parameters. |
| 395 | B | 14 | 3.389189851 | +2.858601% | Localized SS orbital window traded aggregate accuracy against category behavior. |

Useful combinations included density-weighted opposite-spin kinetic averaging, independent or shared OS root responses, density-screened kinetic terms, an SS orbital window, and coefficient sharing. Their effects depended on the surrounding functional and joint fit. Strong SS cancellation gates and several alternate-screen or aggressively pruned variants failed to preserve accuracy.

The important distinction was between **new structural variants** and **same-form fitting controls**. Different training-selected starts sometimes produced different train/validation compromises, including lower training error with worse validation. Those controls count as evaluations, but were not called new symbolic discoveries.

Validation was repeatedly used for search selection. The independent test was therefore essential. Its unfavorable result for candidate 356 is an observed outcome, not a reason to change the frozen candidate, parameters, or shortlist after testing. The results do not uniquely identify the cause of the test regression.

## Frozen shortlist

The following IDs were preregistered **before any test score existed**. The [preregistration](runs/current_session/preregistered_shortlist.json) contains exact source/parameter hashes. The later user instruction narrowed actual testing to the best candidate plus references 1 and 2; it did not change or reorder the original shortlist.

| Order / ID | Parameters | Train | Validation | Gain vs GAS22 | Gap | Validation types >5% worse | Test |
|---|---:|---:|---:|---:|---:|---|---|
| 1. **356** | 14 | 3.207428316 | 3.362805400 | +3.614835% | 0.155377084 | NCD +5.61%, NCED +10.85% | 3.897343411 |
| 2. **395** | 14 | 3.223429879 | 3.389189851 | +2.858601% | 0.165759972 | IE +9.21%, NCED +7.16% | Not evaluated |
| 3. **384** | 14 | 3.198341328 | 3.373369586 | +3.312043% | 0.175028257 | NCD +5.47%, NCED +9.51% | Not evaluated |
| 4. **350** | 15 | 3.189785804 | 3.373554704 | +3.306738% | 0.183768900 | NCED +5.90% | Not evaluated |
| 5. **353** | 16 | 3.185816708 | 3.371371247 | +3.369320% | 0.185554539 | NCED +5.60% | Not evaluated |

All five have zero evaluator violations and pass the numerical SCF gate: validation below 97% of the published ωB97M-V baseline. **Eligibility is not an SCF result.** Candidate 339 is a useful broader-validation comparison, with every validation category within 3% of GAS22, but it was not in the deterministic final five and was not tested.

## Candidate 356: formula and fitted coefficients

Source: [frozen 356.py](runs/current_session/frozen_shortlist/356.py). Evaluated coefficients: [356.params.json](runs/current_session/frozen_shortlist/356.params.json).

**Use the fitted parameter JSON with the source.** The source's `init_params()` preserves historical starting values, not the final evaluated values.

Define:

```text
X_sigma = |grad rho_sigma|^2 / rho_sigma^(8/3)
u_gamma(X) = gamma X / (1 + gamma X)
t_sigma = tau_UEG,sigma / tau_sigma
w_sigma = (t_sigma - 1) / (t_sigma + 1)
u_s = u_0.46914023462026644(X_sigma)
z_sigma = clip(tau_W,sigma / (tau_sigma + 1e-30), 0, 1)
W = [(1 + zeta)/2] w_a + [(1 - zeta)/2] w_b
s = softplus(h) + 1e-12
R = (X_bar + s)^(1/3) - s^(1/3)
V = R / (1 + 0.006 R)
lambda = softplus(q)
```

`X_bar=d.x2_avg` is the API's masked arithmetic mean of squared per-spin reduced gradients. `W` is the density-weighted mean of transformed per-spin kinetic variables, not the default transform of the arithmetic mean of `t`. `r_s` is the total-density Wigner–Seitz radius. Exact masks, kinetic/density guards, and fixed RSH attenuation are in the [formula catalog](runs/current_session/explicit_formulas.json).

```text
g_x,sigma = 0.85 + a_u u_0.003840616724010807(X_sigma) + a_w w_sigma

g_ss,sigma = b0 u_s + b1 w_sigma(1+w_sigma)
           + b2 u_s^6 + b3 u_s^6 w_sigma^4
           + b4 z_sigma(1-z_sigma)

g_os = c0 + c1 W^2(1-W^4) + c2 W^2 V + c3 W^6 V
     + K W(1+W) / sqrt(1+(lambda r_s)^2)
```

Zero-based parameter mapping: `a_u=cx[0]`, `a_w=cx[1]`; `b_j=css[j]` for j=0…4; `c_j=cos[j]` for j=0…3; `h=root_shape[0]`; `q=q[0]`; `K=kinetic_os[0]`. There are **14 trained scalars**.

Only the pure OS `W^2/W^6` pair is tied as `+c1/-c1`; the root-response coefficients remain independent. This is different from shared-root factorization. The full OS factor need not vanish at kinetic endpoints. The quadrature density screen acts only on `K W(1+W)`. The SS orbital window vanishes at its own endpoints but does **not** enforce complete one-electron cancellation of the full SS correlation.

```json
{
  "cx": [
    1.0435360428200626,
    0.20321155891619436
  ],
  "css": [
    0.7274211661294759,
    -4.41901851061404,
    -1.4556670356057029,
    5.735990658405081,
    -0.20458796539440488
  ],
  "cos": [
    1.0538471711342332,
    3.2745582398901556,
    -1.6451114527204427,
    1.7698932792720867
  ],
  "root_shape": [
    0.21548992167904887
  ],
  "q": [
    0.332126814694281
  ],
  "kinetic_os": [
    2.477994532537782
  ]
}
```

Recorded parent chain:

```text
2 -> 16 -> 19 -> 56 -> 79 -> 142 -> 150 -> 236 -> 335 -> 339 -> 350 -> 356
```

Migration 335 also incorporated a C-island idea. A single-parent chain does not capture every migration donor. Some ancestors were flagged stepping stones and repaired in later records; the descendant source was independently reviewed for final eligibility.

## Authorized held-out test

The test set stayed sealed throughout all 400 successes and final selection. The user then explicitly authorized Codex to restart access itself and evaluate the best functional. A fresh MCP child server ran with `DFTSR_ALLOW_TEST=1` for that evaluation; the persistent configuration remained 0.

Exactly one `final_test_evaluation` call was made for **356, 2, and 1**, using stored frozen parameters with `fit=False`. No candidate was refitted or reselected, and no new search records were created. Source and parameter hashes were verified before and after. Recomputed train/validation metrics matched frozen values within 1e-8.

The test-enabled child server and GPU workers were closed afterward. Persistent test access is disabled.

- [Authorization and scope](runs/current_session/test_evaluation_best/authorization.json)
- [Completion and cleanup](runs/current_session/test_evaluation_best/completion.json)
- Raw MCP outputs: [356](runs/current_session/test_evaluation_best/356.json), [GAS22 refit 2](runs/current_session/test_evaluation_best/2.json), [ωB97M-V refit 1](runs/current_session/test_evaluation_best/1.json)

| Frozen record | Parameters | Validation | Test | Validation + test, no RG10 |
|---|---:|---:|---:|---:|
| Candidate 356 | 14 | 3.362805400 | 3.897343411 | 3.691519562 |
| GAS22 refit, seed 2 | 12 | 3.488924272 | 3.627617040 | 3.688485684 |
| ωB97M-V refit, seed 1 | 13 | 3.789175545 | 4.731394656 | 4.135808174 |

| Test type | Reactions | Candidate 356 | GAS22 refit | Improvement vs GAS22 |
|---|---:|---:|---:|---:|
| BH | 40 | 7.293581994 | 6.872114193 | -6.1330% |
| ID | 8 | 22.910350242 | 20.786111763 | -10.2195% |
| IE | 590 | 2.707057370 | 2.780277405 | +2.6336% |
| NCEC | 103 | 5.485778580 | 4.432383222 | -23.7659% |
| NCED | 327 | 3.182306282 | 2.773748679 | -14.7294% |
| TCD | 24 | 1.542995082 | 4.139112294 | +62.7216% |
| TCE | 60 | 2.513845879 | 2.247641633 | -11.8437% |

The ωB97M-V test reference is **the training-refitted seed 1**, not the published-parameter baseline. Positive percentage improvement means lower error. Test category counts sum to 1152. Missing NCD, AE18, and RG10 test observations are not zero-error results.

NCEC and NCED showed the largest relative test regressions: **23.77%** and **14.73%**, respectively. IE and TCD improved, but not enough to offset the aggregate loss.

**V: yes. T: no. Win on both splits: no.** The combined validation+test metric excluding RG10 was also slightly worse than GAS22. This conclusion applies to the one tested candidate; it does not establish test results for the four untested shortlisted records.

## Error handling and integrity corrections

The one failed attempt was **ID 213**, `D054_SharedSSDensityKinetic`: a smoke-test `KeyError: density_os` before fitting. A stale source-rendering closure retained an old parameter reference while the parameter dictionary changed. The failed record was preserved; corrected retry **215** was successful. Errors did not count toward 400 successes.

The earlier renderer issue also affected intended interpretations of records **207** and **210**. Record 207 had an unused parameter and was excluded. Record 210 was relabeled as an initialization control because its intended formula change had not actually been applied. The original records and correction summaries remain in the journal. The helper was repaired, and subsequent source/parameter checks were recorded.

Final verification found **401 matching completed request/database/source pairs**, complete plans and summaries, no missing results, no running evaluations, and per-request fitting limits respected. The final independent audit reviewed 12 leading/control records and retained 29 cumulative exclusions. Synthetic formula and scalar-activity verification files are saved alongside the reports and in each island's directory.

## Compute and token accounting

### Exact locations of per-iteration splits and rationales

All locations below are relative to this project root, `/home/ash/Documents/dft_functional_sr_codex`.

| Location | Contents and exact fields |
|---|---|
| [Concise accounting/rationale index](runs/current_session/iteration_accounting.md) | One entry for each of the **400 successful evaluations**, plus the historical error: all five token fields, concise plan/rationale excerpt, observed outcome, and full-record link |
| [Accounting/rationale CSV](runs/current_session/iteration_accounting.csv) | Same 401 entries; `successful_iteration_number`, `search_attempt_number`, `id`, `island`, `status`, token fields, `tokens_cumulative_run`, `concise_rationale`, `concise_outcome`, `full_record_link` |
| [Complete results CSV](project_iteration_results.csv) | All 404 database records, including three seeds; token splits and concise fields **plus the complete original `plan` and `summary`** |
| [Complete results JSON](project_results.json) | Same fields in `all_iteration_results[]`, as well as the reference/shortlist records; accounting definitions and coverage checks in `per_iteration_accounting` |
| [Authoritative database](runs/iterations.sqlite) | Table `iterations`: `id`, `status`, `plan`, `summary`, `tokens_input`, `tokens_output`, `tokens_cache_read`, `tokens_cache_creation`, `tokens_total` |
| [Full scientific journal](runs/current_session/iteration_reasoning.html) / [JSONL](runs/current_session/iteration_reasoning.jsonl) | Unabridged pre-evaluation plans and post-evaluation summaries; the journal itself is not the token-component export |
| [Accounting verification](runs/current_session/iteration_accounting_verification.json) | Verified coverage, component sums, totals by success/error, provenance, and export timestamp |

The concise rationales are automatically abridged excerpts of the **original pre-evaluation plans**, not new post-test explanations. Ellipses mark truncation. Full plans and summaries remain available through each entry's link. Explicit execution corrections, including records 207 and 210, are retained in the outcome text.

Database IDs are not successful-trial ordinals. The 400 successes span IDs **4–404**, excluding failed ID **213**. `successful_iteration_number` numbers the successes 1–400; the failed attempt has a blank success number and remains separately visible. Seed IDs 1–3 are excluded from the concise search-only index.

| Token field | Meaning |
|---|---|
| `tokens_input` | Uncached input tokens |
| `tokens_output` | Output tokens, including reasoning-output tokens |
| `tokens_cache_read` | Cached input read tokens, separated from uncached input |
| `tokens_cache_creation` | Cache-creation/write tokens; all recorded search values are zero |
| `tokens_total` | Sum of the four components above |
| `tokens_cumulative_run` | Export-derived sum of `tokens_total` over non-seed attempts through the current ID, including the failed attempt |

**Attribution matters:** the server captures a project-wide token delta **at candidate submission**, across all participating Codex sessions since the previous recorded submission. These are tokens *charged to an iteration record*, not an isolated measurement of that island agent's reasoning or of GPU fitting. Concurrent work and summaries of earlier evaluations can fall into a later submission's delta. The database does not store a separate per-iteration reasoning-output count; it is included in `tokens_output` and should not be added again.

The native database field `cum_tokens_total` is a cumulative project-transcript counter, not a reset-based current-run total. Use the exported `tokens_cumulative_run` for this run. No transcript rereading or new evaluation was needed to produce these exports; values came from the stored database fields. Their implementation is documented in [tracker.py](dftsr/tracker.py) and the submission path in [server.py](mcp_server/server.py).

| Charged records | Uncached input | Output | Cache read | Cache creation | Total |
|---|---:|---:|---:|---:|---:|
| 400 successful evaluations | 7,848,847 | 1,026,320 | 512,094,208 | 0 | **520,969,375** |
| One failed attempt, ID 213 | 3,171 | 1,287 | 400,128 | 0 | **404,586** |
| All 401 search attempts | **7,852,018** | **1,027,607** | **512,494,336** | **0** | **521,373,961** |

Every non-seed record has all five stored token values, a nonempty original plan and summary, and a nonempty concise rationale excerpt. For every record, the four components equal `tokens_total`. These checks and totals are saved in the accounting verification file linked above.

To inspect exactly the 400 successful records directly:

```sql
SELECT id, island, tokens_input, tokens_output, tokens_cache_read,
       tokens_cache_creation, tokens_total, plan, summary
FROM iterations
WHERE id > 3 AND status = 'ok'
ORDER BY id;
```

The exports can be regenerated from saved records with `.venv/bin/python scripts/export_iteration_accounting.py`; this performs no scientific evaluation, fitting, or test access.

### Run-level costs and timing

- Summed non-seed worker evaluation time: **30,826.0 seconds** across 401 attempts, approximately **8.563 hours**.
- `search_stats` total including three seeds: **30,962.6 seconds**, averaging **76.6 seconds** across 404 records.
- These are summed worker evaluation durations, not elapsed wall time or pure GPU-kernel profiling. Up to three evaluations ran concurrently.
- Calendar duration and the first resumed submission gap include the **October 19–26 interruption**. The largest recorded submission gap was **613,696.8 seconds**; it must not be labeled active agent reasoning or computation.
- Final validation-report token snapshot: **531,039,091**, including **521,373,961** assigned to iterations and **9,665,130** since the final iteration.
- Assigned components: **7,852,018 uncached input**, **1,027,607 output**, **512,494,336 cached input**, and **0 cache-creation tokens**.

Token accounting includes cached input, not just generated text. These are validation-stage snapshot totals, not an all-inclusive count after the later test and this documentation task. The [structured export](project_results.json) preserves the underlying timing and token fields.

## Environment and reproducibility

The run was orchestrated through Codex with three NVIDIA RTX 6000 Ada GPUs. Exact model/version identity for every resumed agent is not inferred from the repository title alone.

Observed project environment when this document was written:

```json
{
  "python": "3.12.14",
  "packages": {
    "pyscf": "2.14.0",
    "cupy-cuda13x": "14.2.0",
    "gpu4pyscf-libxc-cuda13x": "0.8.1",
    "gpu4pyscf-cuda13x": "1.8.1",
    "mcp": "1.30.0",
    "torch": "2.14.1"
  }
}
```

These are current local observations, not a reconstructed lockfile for every historical evaluation. The `.venv` contains the scientific/MCP tools; the system Python was used for the saved Matplotlib figures.

Useful artifacts and tools:

| Artifact | Purpose |
|---|---|
| [Frozen shortlist and hashes](runs/current_session/preregistered_shortlist.json) | Immutable pre-test selection and reproducibility checks |
| [Frozen source/parameter pairs](runs/current_session/frozen_shortlist/) | Algorithms and separately saved fitted coefficients |
| [Formula catalog](runs/current_session/explicit_formulas.json) | Explicit equations and parameter mappings |
| [Validation trajectory PNG](runs/current_session/validation_search.png) / [PDF](runs/current_session/validation_search.pdf) | All trials, eligible minima, and leading-score zoom |
| [Validation breadth PNG](runs/current_session/validation_breadth.png) / [PDF](runs/current_session/validation_breadth.pdf) | Category-level shortlist tradeoffs |
| [Final A report](runs/current_session/A/checkpoint400_A100_report.json) | Exchange island completion |
| [Final B report](runs/current_session/B/checkpoint400_B120_report.json) | Same-spin island completion |
| [Final C report](runs/current_session/C/C100_report.json) | Opposite-spin island completion |
| [Final D report](runs/current_session/D/final80_report.json) | Parsimony island completion |
| [Test runner](scripts/evaluate_frozen_best_test.py) | Authorized frozen-parameter MCP evaluation with repeat-call guards |
| [Test report generator](scripts/report_best_test.py) | Reporting of saved outputs; no fitting or evaluation |
| [Provenance verifier](scripts/verify_current_run.py) | Request/source, documentation, and budget checks |

Current state: **search frozen; candidate 356 and references 1/2 tested once; other shortlisted candidates untested; test access disabled; zero SCF jobs**. Do not rerun completed tests, refit or reselect using held-out outcomes, or describe a later search as part of this untouched held-out evaluation. SCF or a genuinely new experiment would require a separate explicit decision and should not be represented as already completed.
