# Forensic report: recurring model-version state management

**Source run:** `results/model_version_aware_recurring_state_management_full_20260827T134710Z`  
**Scope:** read-only analysis of the completed 10-seed run. No experiment, source, preregistration, checkpoint, or existing result artifact was modified.

## Artifact verification

- Required root artifacts are present: provenance, resolved configuration, command record, final comparison, and paired Phase-C/Phase-D CSVs.
- All 10 seeds and both policies have `episodes.csv` and `boundary_state_audit.json`: 20 of each file.
- Every policy/seed CSV has exactly 100 records in each of A, B, C, and D.
- Direct CSV comparison confirms exact equality for every seed in both A and B after excluding only policy/seed and runtime/timestamp metadata.
- The per-seed Phase-D CSV reconstruction reproduces the reported values: mean reward difference `+0.000476682978`; sample SD `0.006487751926`; median `+0.001114471642`; range `[-0.008992293987, +0.009700457378]`; 5 positive, 5 negative, 0 ties. Mean isolation-minus-reset differences are `-0.001` recovery, `-0.001` defended accuracy, and `-0.000198109346` post-risk.

See `forensic_summary.json`, `raw_equality_and_phase_counts.csv`, and `phase_d_raw_reconstruction.csv`.

## Restoration verification

For every seed, the robust learner-state hash saved as the departing state at B→C is exactly equal to the version-isolation active state hash at D. All D isolation boundaries report `restored_existing_version`, matching restore hashes, both registry keys, and passed integrity. All full-reset D boundaries report `fresh_full_reset` with B-at-prior, f-at-zero, and regenerated-cache checks passing.

The implemented persisted-state interface covers B/f sufficient statistics, learner RNG state, serialized CUSUM state, update count, cumulative reward, per-arm aggregates, and timestamps; cache is regenerated from B/f on `load_state`. The audits establish hash equality of that complete serialized state and separate fresh-prior/cache checks. They do not retain numeric B/f matrices or raw RNG/CUSUM payloads, so matrix-norm distances or direct field-by-field post-hoc comparisons cannot be calculated from these artifacts.

No evidence indicates state sharing: inactive registry hashes are reported unchanged during every active phase, and each registry contains distinct `standard` and `robust_alone` keys.

## Phase-D trajectory analysis

Restoration changed behavior immediately: the first selected-arm difference occurred at D episode 1 for six seeds and episode 2 for four. Policies selected the same arm in only 26–46 of 100 D episodes per seed (mean 33.9); therefore the treatment was behaviorally active, not a hidden tie.

The initial ten D episodes had a mean isolation-minus-reset reward difference of `+0.005145`, but this varied from `-0.032831` to `+0.030944`. The last 90 episodes averaged `-0.000042`, varying from `-0.011883` to `+0.009425`. The correlation between each seed's first-10 difference and its full-D difference was `0.022`, so early advantage did not predict final advantage. The episode-level reward, risk, selected-arm, posterior projection/uncertainty, recovery, accuracy, and update fields are retained in the source CSVs; the reconstruction table provides the non-cherry-picked per-seed summaries.

## Explaining the 5–5 split

There is no stable observable pattern supporting a reliable D advantage. The Phase-D reward signs are mixed across the entire seed range. Phase-C reward difference is positive in 8/10 seeds, but its correlation with Phase-D reward difference is only `0.317` and it is a secondary endpoint. Phase-D arm-divergence count has correlation `-0.117` with D reward difference; early-D reward has correlation `0.022`; post-risk difference has correlation `-0.048`. These descriptive correlations are not inferential claims; they show no obvious deterministic explanation in the retained outputs.

Recovery/accuracy differences have a stronger descriptive association with D reward difference (`0.705`), which is expected because they are components of the same realized defense outcomes. It does not establish causality or rescue the failed preregistered recovery/accuracy gates.

## Bookkeeping and methodology

The 100 versus 200 D update totals are expected. Full reset enters D at update 0 and leaves at 100. Isolation restores the 100-update robust B state and leaves D at 200. This is the intervention, not contamination. D primary reward was recomputed solely from the 100 D `reward` records, not from historical `cumulative_reward_after`; consequently the inherited cumulative-reward/aggregate counters do not enter the paired D mean comparison.

The test fairly measures the stated complete-state restoration policy, including retained RNG, CUSUM, and aggregate state. It cannot isolate a posterior-only benefit from the benefit or cost of restoring the complete learner state, because complete-state restoration is the preregistered intervention. No artifact evidence indicates a restoration, measurement, manifest, equality-gate, checkpoint, or update-progression defect.

## Final diagnosis

**A. Result appears valid: genuine weak/null/sign-varying effect.** Restored B state is demonstrably loaded and changes D choices promptly, but its realized Phase-D reward effect is small relative to seed-to-seed variation, has no reliable sign, and fails the preregistered consistency and recovery/accuracy requirements. No statistical-significance claim is made.

## Single next step

Do not rerun or tune this protocol. Record this as a valid negative result and conduct no further experiment until a separately preregistered question is justified. If future work is pursued, the question should specifically be whether *complete-state* restoration's non-posterior components (restored RNG/CUSUM/aggregates) materially alter recurrence outcomes relative to restoring only the mathematically sufficient B/f posterior state; the current study cannot answer that decomposition because it correctly tested the full registered state as one intervention.
