# Model-Version-Aware Adaptive DA-LinTS State Management

## 1. Title

**Model-Version-Aware Adaptive DA-LinTS State Management: a pre-registered paired comparison at the `standard` to `robust_alone` boundary**

## 2. Research question

When an explicitly observed classifier/model version changes, does version-indexed DA-LinTS state isolation improve Phase-B adaptive-defense performance relative to the validated full Phase-B reset?

The primary regime-change signal is the observed `environment_model` / model-version boundary, not CUSUM.

## 3. Motivation and existing evidence

The matched three-seed continuous-state comparison found lower Phase-B reward and higher post-defense risk for continuous state on all three seeds. It did not establish statistical significance, but it rules out a claim that stale posterior preservation is beneficial in this setting. The pre-registered 10-seed fixed 50% `B`/`f` shrinkage comparison did not justify replacing full reset; that conclusion is treated as established.

Inspection of `adaptive/bandit.py` establishes that the DA-LinTS posterior state is three per-arm sufficient-statistic pairs `(B_a, f_a)`. Posterior mean, covariance, Cholesky factor, and uncertainty are deterministic caches rebuilt from those statistics. The serialised learner additionally contains the RNG state, CUSUM detector state, update/reward aggregates, arm aggregates, and timestamps. Thus `B` and `f` are the only learned quantities that can legitimately carry reward-model information across a version boundary; cached quantities must never be transferred independently.

The current transition has no pre-existing `robust_alone` learner state. Consequently, a scientifically defensible version-aware policy must create the robust-state partition at its prior, rather than manufacture a transferred posterior or choose another arbitrary shrinkage constant. This preregistration intentionally tests that disciplined state lifecycle. It also makes clear that, on this first one-way transition, its active Phase-B posterior is expected to be behaviorally equivalent to full reset when random streams are coupled. A failure to exceed full reset is therefore an informative, valid negative result, not a reason to tune a transfer amount.

## 4. Hypotheses

**Primary hypothesis (H1).** Under the fixed matched protocol, version-indexed state isolation will have a positive paired Phase-B mean-reward difference versus full reset while meeting all safety and integrity gates below.

**Null/negative hypothesis (H0).** The intervention does not satisfy every advancement gate. In particular, because `robust_alone` has no pre-existing state at this boundary, it may tie full reset exactly; any such tie or degradation means full reset remains the baseline. No statistical-significance claim will be made solely from these 10 seeds.

## 5. Policies

### Baseline: full Phase-B reset

Run one ordinary local three-arm DA-LinTS learner during Phase A. Immediately before Phase B, replace it with a fresh learner using the unchanged constructor configuration and the Phase-B sampling stream defined in the manifest/RNG protocol. Its posterior starts at `B_a = lambda I` and `f_a = 0` for every arm `a`; CUSUM and aggregate diagnostic state are fresh as in the existing validated reset baseline.

### Intervention: version-indexed learner-state isolation

Maintain an experiment-local registry keyed by the exact model-version identifier. At the boundary, freeze and retain the complete `standard` learner state under key `standard`; then activate key `robust_alone`.

If an active key exists, restore that key's complete learner state. If it does not exist, as it will not for `robust_alone` in this experiment, initialise a fresh state at the ordinary DA-LinTS prior. The active learner alone selects arms and receives updates. The inactive `standard` state is read-only and is neither sampled, merged, discounted, nor updated in Phase B. A per-version CUSUM instance uses the unchanged existing configuration.

This is exactly one intervention: explicit model-version partitioning and controlled state invalidation. It is not a partial reset, a parameter sweep, an ensemble, or a new learning rule.

## 6. Mathematical state definition and boundary timing

Let `V = {standard, robust_alone}` and let each version state contain, for arm `a in {light, medium, strong}`,

`L_v = ({B_{v,a}, f_{v,a}}_a, R_v, D_v, G_v)`,

where `R_v` is the learner RNG state, `D_v` is the unchanged CUSUM detector state, and `G_v` is the existing scalar/arm diagnostic state. Derived posterior caches are not stored as independent state: `mu_{v,a}=B_{v,a}^{-1}f_{v,a}` and `Sigma_{v,a}=sigma^2 B_{v,a}^{-1}` are regenerated from `B` and `f`.

At the exact transition after Phase-A episode 100 and before loading/running the first Phase-B episode:

`registry[standard] <- L_standard^(100)`

`L_robust_alone^(0) = ({lambda I, 0}_a, R_B, D_0, G_0)`

`active <- L_robust_alone^(0)`.

`R_B` is predeclared so the intervention and full-reset policies use the same Phase-B Thompson-sampling random stream for a seed. `D_0` and `G_0` are the ordinary fresh-state values under the unchanged implementation/configuration. No value from `L_standard` is inserted into the active `B`, `f`, cache, CUSUM, RNG, or diagnostic state. Standard state is retained solely as a provenance-preserving inactive version partition. Therefore the active Phase-B posterior prior is exactly the full-reset prior, with no arbitrary coefficient.

The ordinary DA-LinTS update remains exactly `B_a <- B_a + phi phi^T` and `f_a <- f_a + phi r`; equations, actions, rewards, risk, defenses, and thresholds are unchanged.

## 7. Fixed experimental protocol

- Seeds, pre-registered in advance: `20260821, 20260822, 20260823, 20260824, 20260825, 20260826, 20260827, 20260828, 20260829, 20260830`.
- Phase A: 100 episodes, `environment_model=standard`, `checkpoints/cnn_best.pth` (SHA-256 `e548c7a95c030a206f5b1da6620209ef67b77dee90f1fe33b8200a3deb09caf8`).
- Phase B: 100 episodes, `environment_model=robust_alone`, `checkpoints/robust_best.pth` (SHA-256 `cf518eff5c283a967d376b12d0765daa4c5d82dc3d5f9d681b5a9d892299c4d3`).
- Boundary: `standard -> robust_alone`, exactly once between episodes 100 and 101.
- Actions: unchanged `light`, `medium`, `strong`.
- Each seed receives one fixed 200-episode manifest generated before either policy runs. The same manifest, attack schedule, image indices, attack types, epsilons, torch randomness, and policy sampling stream are shared by both policies for that seed. The matched Phase-A trajectories must be identical; Phase-B equality is expected under this specified first-exposure intervention and must be audited rather than assumed.
- Keep unchanged: DA-LinTS equations and hyperparameters; action space; reward definition; risk computation; attack generation and severity; defenses; thresholds; classifier checkpoints; evaluation procedure; logging methodology; CUSUM implementation/configuration; existing posterior-discount implementation; dataset handling; and runtime environment.

## 8. Metrics and reporting

The primary per-seed quantity is `d_s = mean_reward_B(intervention, s) - mean_reward_B(full_reset, s)`. Report each seed's full-reset Phase-B reward, intervention Phase-B reward, and `d_s`; then report the mean, sample standard deviation, minimum, maximum, positive count, negative count, and tie count of `d_s`.

Secondary Phase-B metrics, reported per seed and in distributional/aggregate form for each policy, are recovery rate, defended accuracy, mean pre-defense risk, mean post-defense risk, risk reduction, selected-arm counts/proportions, posterior diagnostics (per-arm posterior projection and uncertainty before/after updates plus boundary-state verification), CUSUM events, posterior-discount events, active-learner update counts, and runtime. Episode-level records, not only aggregates, are required.

For a metric where lower is better (post-defense risk), report intervention minus full reset and state the direction explicitly. Ties are values exactly equal in the stored numeric output; no rounded display value determines a tie.

## 9. Preregistered advancement and failure rules

The intervention is considered better than full reset only if **all** of the following hold:

1. `mean_s(d_s) > 0`.
2. At least 8 of the 10 seeds have `d_s > 0`.
3. Mean Phase-B post-defense risk for the intervention is no higher than full reset: `mean(post_risk_intervention - post_risk_full_reset) <= 0`.
4. There is no material degradation: mean Phase-B recovery difference and mean Phase-B defended-accuracy difference are each `>= 0`. This zero-tolerance rule is deliberately conservative.
5. No implementation-integrity or pathological-detector failure occurs: the manifests/checkpoint hashes/configuration are matched; exactly 100 active Phase-B updates occur per policy per seed; the intervention's active Phase-B `B`/`f` equal the prior at the boundary and contain no standard-state contribution; detector settings and posterior-discount path are unchanged; and all artifacts validate.

If any condition fails, full Phase-B reset remains the baseline. A negative result, including an exact tie expected from a new `robust_alone` partition, is a valid research result. These rules do not constitute a significance test and must not be described as statistical significance.

## 10. CUSUM's scientific role

CUSUM remains an unchanged secondary diagnostic. Before each update it receives the existing standardized residual

`e_t = (r_t - mu_a^T phi_t) / sqrt(sigma^2 + phi_t^T Sigma_a phi_t)`.

It retains `kappa=0.5`, threshold `5.0`, startup/cooldown behavior, and on-alarm `forgetting_factor=0.95` discount over every active arm's real `B` and `f`. CUSUM events and posterior discounts are logged. They are not required to occur, are not used as the model-version signal, and must not be interpreted as CUSUM detecting the model-version change. The prior transition validations' zero events/discounts remain correct evidence that this reward-residual CUSUM is not a primary known-version detector.

## 11. Reproducibility and artifact requirements

Create a new, non-overwriting research-results directory containing the immutable preregistration copies, resolved config, exact seed list, manifests and SHA-256 hashes, checkpoint hashes, source-file hashes/revision metadata, dependency/runtime metadata, and command record. For every seed and policy retain episode-level CSV/JSON records, a boundary-state audit (active/inactive version keys, B/f prior checks, posterior-cache regeneration check, detector config/state, RNG-stream protocol), policy summary, and runtime.

Create a paired per-seed table and a final comparison summary with every required primary/secondary metric, exact advancement-gate booleans, and a statement that no significance claim follows from 10 seeds. Preserve raw artifacts, logs, manifests, and summaries even when unfavorable. The analysis must be deterministic from the retained artifacts.

## 12. Prohibitions after preregistration

The following are forbidden after results are observed: post-hoc tuning; changing/adding reset percentages; parameter sweeps; selecting an intervention based on results; changing rewards, risk, attacks, attack severity, defenses, thresholds, DA-LinTS equations, action space, classifier checkpoints, evaluation or logging; changing CUSUM `kappa`, threshold, startup/cooldown behavior, or discount factor; cherry-picking or selectively rerunning seeds; deleting, overwriting, or hiding unfavorable artifacts; and changing the advancement criterion.

No new attack transition may be invented to make CUSUM alarm. No production code or Streamlit application is within the scope of this preregistration. If an implementation detail later conflicts with this specification, record the uncertainty and stop for an amendment before execution; do not guess or silently alter the design.

## 13. Expected scientific contribution

This experiment separates an observed model-version event from residual drift detection and establishes a provenance-safe state-management policy. It tests whether version-aware isolation can outperform the validated reset baseline under the fixed transition, while preventing ungrounded stale-posterior transfer. Its most likely first-transition outcome—equivalence to full reset—is still useful: it demonstrates that version partitioning is a lifecycle/provenance mechanism, not evidence for invented transfer benefits. Future recurring-version work would require a separately preregistered design with an already accumulated state for the returning version.
