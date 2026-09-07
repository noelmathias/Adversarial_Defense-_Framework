# Preregistration: Recurring Model-Version-Aware Adaptive DA-LinTS State Management

## Question and rationale

The completed one-way `standard -> robust_alone` study tied exactly because
`robust_alone` had no previously accumulated learner state.  This new,
separate study tests the actual lifecycle claim: whether version-indexed
isolation correctly preserves and restores a complete learner state when an
already observed model version returns.  It neither revises nor reruns the
completed study.

## Fixed protocol

- Seeds: `20260821` through `20260830`, inclusive.
- Per seed: one fixed 400-episode manifest, generated before either policy.
- Four 100-episode phases: A=`standard`, B=`robust_alone`, C=`standard`,
  D=`robust_alone`.
- `standard` uses `checkpoints/cnn_best.pth` and `robust_alone` uses
  `checkpoints/robust_best.pth`.  Their required SHA-256 values are,
  respectively, `e548c7a95c030a206f5b1da6620209ef67b77dee90f1fe33b8200a3deb09caf8`
  and `cf518eff5c283a967d376b12d0765daa4c5d82dc3d5f9d681b5a9d892299c4d3`.
- Actions remain `light`, `medium`, `strong`.  DA-LinTS, reward, risk,
  attacks, attack severities, defenses, thresholds, dataset handling,
  evaluation/logging fields, CUSUM configuration, and posterior-discount
  behavior are unchanged.
- Torch episode seeding remains `seed + manifest episode`.  A newly created
  learner is seeded with the seed, as in the completed study.  A restored
  version partition restores its complete persisted learner state, including
  its RNG and CUSUM/aggregate state; no independent cache is transferred.

## Policies and boundary mechanics

`full_reset` creates an unchanged fresh local learner at every boundary.

`version_indexed_learner_state_isolation` owns an experiment-local registry
keyed by the exact identifiers `standard` and `robust_alone`.  At every
boundary it snapshots the departing active learner under its version key.
For a missing entering key it creates the ordinary fresh prior.  For an
existing entering key it restores that complete snapshot.  Inactive registry
states are never sampled, updated, merged, discounted, or otherwise changed.

Thus B is a first visit to `robust_alone`, C restores the state learned in A,
and D restores the state learned in B.  The active DA-LinTS updates are still
exactly the existing implementation's updates; this study adds no transfer
coefficient, ensemble, or new learning rule.

## Hypothesis and endpoint

The primary per-seed outcome is

`d_s = mean_reward_D(version_indexed_learner_state_isolation, s) - mean_reward_D(full_reset, s)`.

The primary hypothesis is that restored `robust_alone` state produces a
positive mean Phase-D paired reward difference without higher mean Phase-D
post-defense risk or degradation in Phase-D recovery or defended accuracy.
Phase C is a predeclared secondary recurrence endpoint.  No significance
claim is made from these ten seeds.

## Integrity and advancement decision

Phase A trajectories must be exactly equal between policies before B begins.
Phase B trajectories must also be exactly equal before C begins, because
both policies enter the first `robust_alone` exposure at the same fresh
prior and sampling seed.  A mismatch stops that seed immediately and
preserves artifacts.  C and D are expected to differ and are not equality
gates.

For every boundary, retain state hashes, update counts, prior/cache checks
where an entering state is new, and restoration/hash/unchanged-inactive-state
checks where it is returning.  The Phase-D evidence decision is favorable to
version-indexed isolation only if all hold:

1. mean `d_s > 0`;
2. at least 8 of 10 `d_s > 0`;
3. mean Phase-D post-risk difference (isolation minus full reset) is `<= 0`;
4. mean Phase-D recovery and defended-accuracy differences are each `>= 0`;
5. all manifest, checkpoint, equality-gate, boundary, update-count, and
   artifact-integrity checks pass.

Otherwise `full_reset` remains the baseline.  This is an evidence decision,
not a statistical-significance result.

## Artifact and provenance requirements

Each non-overwriting results directory retains copies of this preregistration,
resolved config, manifest and hashes, checkpoint and source hashes, runtime
provenance, command record, episode CSVs, per-boundary audits, policy
summaries, paired Phase-D and secondary Phase-C comparisons, and the final
decision.  Existing experiments, results, preregistrations, production code,
Streamlit, and checkpoints are out of scope.
