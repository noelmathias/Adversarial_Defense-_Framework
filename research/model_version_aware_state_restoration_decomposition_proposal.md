# Proposed follow-up preregistration draft: recurring learner-state restoration decomposition

**Status:** proposed design only — not approved, implemented, or executed.  
**Relationship to completed work:** this is a separate follow-up to the closed
`model_version_aware_recurring_state_management` experiment. It does not amend,
rerun, reinterpret, or replace that experiment or its diagnostics.

## Question

When `robust_alone` recurs, are behavioral differences attributable to the
learned DA-LinTS sufficient posterior state (`B`, `f`), or to restoring the
additional complete-state fields (learner RNG, CUSUM detector state, global and
per-arm aggregates, and timestamps)?

The completed study established that complete-state restoration was correct,
behaviorally active, and weak/sign-varying relative to reset.  It did not
identify which state component caused those differences.

## Fixed candidate protocol

- Inherit seeds `20260821`–`20260830`.
- One fixed 400-episode manifest per seed, written before any policy runs.
- Four 100-episode phases: A=`standard`, B=`robust_alone`, C=`standard`,
  D=`robust_alone`.
- Reuse the exact existing checkpoints, action space (`light`, `medium`,
  `strong`), attacks, epsilons, reward/risk/defense procedures, CUSUM
  configuration, posterior-discount path, dataset handling, episode Torch
  seeding, and CSV logging semantics.
- Retain the classifier checkpoint hashes and all runtime/provenance capture
  requirements from the prior recurring study.

## Policies

All policies use experiment-local state only; production state is never read
or written.

| Policy | State on every boundary |
| --- | --- |
| `full_reset` | New unchanged learner: `B=lambda I`, `f=0`, fresh learner RNG seeded by the predeclared seed, fresh CUSUM, zero global/per-arm aggregates. |
| `posterior_state_restoration` | Snapshot only `B` and `f` under each exact version key. On recurrence, create a fresh unchanged learner and replace only its `B`/`f` with the stored matrices/vectors; regenerate derived posterior cache through existing `load_state`. Keep fresh learner RNG, fresh CUSUM, zero global/per-arm aggregates, fresh timestamps, and frozen fresh-configuration hyperparameters. |
| `complete_state_restoration` | Snapshot and restore the complete existing serialized learner state: `B`, `f`, hyperparameters, learner RNG, CUSUM state, update count, cumulative reward, per-arm aggregates, and timestamps. Derived posterior cache is regenerated from `B`/`f`, never independently copied. |

At a missing version key (B on first `robust_alone` exposure), both restoration
policies initialise the unchanged fresh prior. At C and D, each restoration
policy restores its own prior state accumulated for the returning version.
Inactive version states remain read-only: never sampled, updated, merged, or
discounted.

`posterior_state_restoration` is deliberately a *joint contrast* against
complete restoration: complete-minus-posterior captures the combined impact of
the restored RNG, detector, aggregates, and timestamps. It does not identify
the separate causal contribution of each non-posterior field.

### Exact posterior-only construction

At each returning-version boundary, the research adapter must construct the
posterior-only active learner exactly as follows:

```text
fresh_state = fresh_bandit(seed).get_state()
posterior_state = copy(fresh_state)
posterior_state["B"] = copy(saved_version_state["B"])
posterior_state["f"] = copy(saved_version_state["f"])
active.load_state(posterior_state)
```

The adapter must not copy from `saved_version_state` any RNG state, detector
state, update count, cumulative reward, arm statistics, timestamps, or
hyperparameters. Hyperparameters are those already present in `fresh_state`;
they must match the frozen configuration and be audited. This construction is
the sole posterior-only intervention.

### Definition of complete state

For this study, **complete state** means exactly the payload currently emitted
by `ContextualBandit.get_state()`: B/f; context/action/hyperparameters;
learner RNG state; `drift_enabled` and serialized detector state; update count,
cumulative reward, and per-arm aggregate statistics; and `created_at` /
`updated_at` timestamps. Derived posterior caches are not serialized and are
regenerated through `load_state()`.

`stats.history` and `last_drift_diagnostic` are not in the current
`get_state()` payload, so they are not part of the complete-state intervention.
Inspection of the current implementation confirms that history is append-only
diagnostic data and that `last_drift_diagnostic` is reset by `load_state()`;
neither is read by action selection or the posterior update equations.

## Random-stream control

At D, `full_reset` and `posterior_state_restoration` instantiate fresh learners
with the same predeclared per-seed constructor RNG state. Therefore their
selection-stream difference can arise only from `B`/`f` (given matched inputs).
`complete_state_restoration` intentionally restores the B-phase RNG as part of
the complete-state intervention; its contrast with posterior-only includes
that restored RNG, as preregistered. Torch episode randomness remains fixed by
`seed + manifest_episode` for every policy.

## Estimands and frozen evidence decision

The primary endpoint remains D episode-level mean reward per seed. Predeclare
three paired contrasts, always computed as left-minus-right in the named order:

1. posterior-only minus reset — effect attributable to restoring learned
   sufficient posterior state under fresh non-posterior state;
2. complete-state minus posterior-only — incremental effect of restoring the
   joint non-posterior state;
3. complete-state minus reset — total complete-restoration effect, retained
   for continuity with the completed study.

For each contrast report D reward, recovery, defended accuracy, pre/post risk,
risk reduction, selected-arm counts, posterior diagnostics, CUSUM events,
posterior discounts, and runtime. Phase C is secondary and is never promoted
to the primary endpoint.

The **primary causal contrast** is Phase-D
`posterior_state_restoration - full_reset`. It is the only contrast that may
support a policy/implementation decision. Using the same pre-existing,
conservative rule used for the prior state-management studies—not a threshold
selected from the completed recurring result—posterior-only restoration is
considered evidence sufficient to replace the reset baseline only if all hold:

1. mean per-seed Phase-D reward difference is `> 0`;
2. at least 8 of 10 per-seed Phase-D reward differences are `> 0`;
3. mean Phase-D post-risk difference is `<= 0`;
4. mean Phase-D recovery and defended-accuracy differences are each `>= 0`;
5. all manifest, checkpoint, first-visit equality, boundary-component, and
   artifact-integrity gates pass.

If any condition fails, full reset remains the baseline. No statistical
significance claim is authorized from these ten seeds.

The **prespecified decomposition contrast** is Phase-D
`complete_state_restoration - posterior_state_restoration`. It is reported
with the same paired reward and safety distributions, but is descriptive and
cannot by itself authorize a policy/implementation decision. Its purpose is
component attribution: it estimates the joint incremental effect of restoring
the non-posterior serialized state. The total
`complete_state_restoration - full_reset` contrast is descriptive continuity
reporting only.

## Integrity gates and audits

- A must be exactly equal across all three policies before B; B must be
  exactly equal across all three policies before C. Equality excludes only
  policy identity, timing, and timestamp metadata. A substantive mismatch
  stops immediately with artifacts preserved.
- Audit each boundary with policy, entering/departing version, state mode,
  manifest/checkpoint hashes, update counts, component hashes, CUSUM
  configuration, and inactive-state immutability.
- A single full-state hash is insufficient to prove posterior-only restoration.
  For posterior-only restoration retain separately: source B/f hash and values;
  applied B/f hash and exact equality; fresh-base full-state hash; fresh-base
  non-posterior-component hash; active non-posterior-component hash and exact
  equality to fresh base; RNG state hash; complete serialized CUSUM state and
  configuration hash; `n_updates`; cumulative reward; every arm aggregate;
  timestamps; frozen hyperparameter equality; and regenerated posterior-cache
  consistency.
- For complete restoration retain separately: complete saved-state hash and
  complete restored-state hash with exact equality; B/f hash; RNG state hash;
  complete serialized CUSUM state/configuration hash; `n_updates`; cumulative
  reward; every arm aggregate; timestamps; and regenerated cache consistency.
- For reset audit: B/f prior checks, fresh RNG/CUSUM/aggregates, and cache
  consistency.

At D start, audits must additionally prove that (i) reset and posterior-only
have identical fresh RNG-state hashes, (ii) posterior-only has a fresh CUSUM
state and zero aggregates, (iii) complete restoration has exactly the stored
B-phase RNG/CUSUM/aggregate state, and (iv) posterior-only and complete
restoration have identical B/f hashes because both derive from the same
first-visit B state.

Retain per-policy episode CSVs, phase CSVs, raw state-component audit JSON,
paired tables for each contrast, source/revision/dependency/runtime metadata,
and non-overwriting output directories.

## Confounding risks and limits

1. `B`/`f` restore alters posterior mean and uncertainty while fresh RNG keeps
   the sampling stream common to reset; this is the intended posterior test.
2. Complete-minus-posterior bundles RNG, CUSUM, all aggregates, and timestamps.
   It cannot say which one is responsible. Timestamps are retained only because
   they are part of the current complete serialized state and are not expected
   to influence selection.
3. CUSUM can discount B/f during D. Thus the complete-minus-posterior contrast
   can subsequently affect posterior state through detector history; that is a
   causal consequence of complete-state restoration, not a posterior-only
   effect.
4. The three policies diverge after first recurrence, so later matched episode
   outcomes share manifests and Torch randomness but not actions/rewards. This
   is the intervention effect, not a violation of matching.
5. Ten inherited seeds support the continuity protocol but do not justify a
   statistical-significance claim. Descriptive distributions and any frozen
   evidence criteria must be interpreted accordingly.

## Required approval before implementation

Approval must settle whether to authorize implementation of this frozen design.
No code or experiment should begin from this draft alone.
