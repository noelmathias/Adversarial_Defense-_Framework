# Proposed analysis plan: state-restoration decomposition

**Status:** design support only; no implementation or execution is authorized.

## Input validation

For every seed verify the single 400-episode manifest, checkpoint hashes,
runtime provenance, 100 records per A/B/C/D phase per policy, and all three
policy directories. Validate A/B raw-record equality pairwise for reset vs
posterior-only, reset vs complete, and posterior-only vs complete.

## Boundary verification

At B verify all policies are fresh-prior equivalent. At C and D verify:

- reset starts at prior with fresh RNG, detector, and aggregates;
- posterior-only B/f equal the saved state for the entering version exactly,
  while its non-posterior fields equal its fresh-base state exactly;
- complete state equals its complete saved source state exactly; and
- inactive-version snapshots remain unchanged while another version is active.

Audits must separately hash B/f and the full serialized state. A full-state
hash alone cannot establish posterior-only restoration. For posterior-only,
retain and compare source/applied B/f, fresh-base/active non-posterior
components, RNG, complete CUSUM state/configuration, update count, cumulative
reward, every arm aggregate, timestamps, frozen hyperparameters, and
regenerated cache consistency. For complete restoration, retain matching
component hashes and complete saved/restored equality.

At D explicitly verify that reset and posterior-only have identical fresh RNG
states; posterior-only has fresh CUSUM and zero aggregates; complete
restoration exactly matches stored B-phase RNG/CUSUM/aggregates; and
posterior-only and complete restoration have identical B/f hashes.

## Primary D tables

Create one per-seed paired row for each predeclared contrast with:

- mean D reward, recovery, defended accuracy, pre/post risk, risk reduction;
- selected-arm counts and proportions;
- first selected-arm divergence episode versus reset;
- mean reward difference for D episodes 1–10 and 11–100;
- D update counts before/after; and
- CUSUM/discount event counts.

Summarize each contrast with mean, median, sample SD, min, max, positive,
negative, and tie count. Episode-level records are the source of truth; summary
JSON is checked, not trusted.

## Interpretation limits

Posterior-only minus reset is evidence about restoring B/f under fresh
non-posterior state. Complete minus posterior-only is evidence about the
combined effect of restored RNG/CUSUM/aggregates, not each component
individually. The frozen primary causal contrast is posterior-only minus reset.
It can support replacing reset only if all inherited conservative criteria
pass: positive mean D reward difference, at least 8/10 positive D seeds, no
higher mean post-risk, nonnegative mean recovery and defended accuracy, and
all integrity gates. Complete-minus-posterior and complete-minus-reset are
descriptive only and cannot independently authorize implementation. No
significance claim is made from 10 seeds. Phase C is reported as secondary
only.
